# app.py
# Personal Finance Chatbot: Intelligent Guidance for Savings, Taxes, and Investments
# UI: Streamlit (single-file), Providers: Local Llama 3 (Ollama), IBM watsonx.ai (Granite), Hugging Face
# Run:  streamlit run app.py

import os
import json
import math
import time
import re
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import streamlit as st

# ------------------------------
# Page Setup & Theming
# ------------------------------
st.set_page_config(
    page_title="FinBuddy • Smart Savings, Taxes, Investments",
    page_icon="💸",
    layout="wide",
)

# Minimal glassmorphism vibes + neon accents
CUSTOM_CSS = """
<style>
:root {
  --glass-bg: rgba(255, 255, 255, 0.5);
  --glass-border: rgba(255, 255, 255, 0.35);
}

/* Background gradient */
[data-testid="stAppViewContainer"] > .main {
  background: radial-gradient(1200px 600px at 10% 10%, #e3f2fd 0%, transparent 60%),
              radial-gradient(800px 400px at 90% 30%, #fce4ec 0%, transparent 60%),
              linear-gradient(180deg, #fafafa 0%, #f5f7fb 100%);
}

/* Glass cards */
.block-container {
  padding-top: 1rem !important;
}

.glass {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  box-shadow: 0 10px 30px rgba(0,0,0,0.06);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-radius: 18px;
  padding: 1rem 1.25rem;
}

.badge {
  display: inline-flex; align-items: center; gap: .45rem;
  padding: .25rem .6rem; border-radius: 999px; font-size: .8rem;
  border: 1px solid #e0e0e0; background: black; box-shadow: 0 2px 8px rgba(0,0,0,.04);
}

.kpi {font-size: 2rem; font-weight: 700; margin: 0;}
.kpi-sub {color: #616161; font-size: .9rem; margin-top: .25rem;}

.chat-bubble-user {background:blue; padding: .75rem 1rem; border-radius: 16px;}
.chat-bubble-ai   {background: black; padding: .75rem 1rem; border-radius: 16px;}

/* Pretty sliders */
.css-10trblm, .stSlider, .stNumberInput > div > div > input {font-weight: 600;}

/* Sticky footer toolbar */
.toolbar {
  position: sticky; bottom: 0; padding: .75rem; border-top: 1px dashed #ddd;
  background: rgba(255,255,255,.75); backdrop-filter: blur(6px);
}

.small {font-size: .85rem; color: #616161}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------
# Utilities
# ------------------------------
@dataclass
class Msg:
    role: str
    content: str


def _pii_redact(text: str) -> str:
    """Very light-touch redaction for obvious PII (demo only)."""
    patterns = [
        (r"\b\d{12}\b", "[AADHAAR_REDACTED]"),
        (r"\b\d{16}\b", "[CARD_REDACTED]"),
        (r"\b\d{10}\b", "[PHONE_REDACTED]"),
        (r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", "[EMAIL_REDACTED]"),
        (r"IFSC[:\s]*[A-Z]{4}0[0-9A-Z]{6}", "IFSC:[REDACTED]"),
    ]
    for pat, repl in patterns:
        text = re.sub(pat, repl, text)
    return text


def _ensure_state():
    if "chat" not in st.session_state:
        st.session_state.chat: List[Msg] = []
    if "savings_plan" not in st.session_state:
        st.session_state.savings_plan = {}


def _toast(msg: str, icon: str = "✅"):
    st.toast(f"{icon} {msg}")


# ------------------------------
# Provider backends
# ------------------------------

# 1) Local Llama 3 via Ollama

def call_ollama(model: str, messages: List[Dict[str, str]], host: Optional[str] = None, temperature: float = 0.2) -> str:
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature}
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "(No content)")
    except Exception as e:
        return f"[Ollama error] {e}"


# 2) IBM watsonx.ai (Granite)
#    Requires: pip install ibm-watsonx-ai
#    Set env: WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID

def call_watsonx_granite(prompt: str, model_id: str = "ibm/granite-13b-instruct-v2") -> str:
    try:
        from ibm_watsonx_ai.foundation_models import Model
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        from ibm_watsonx_ai import Credentials
    except Exception:
        return "[watsonx.ai SDK missing] pip install ibm-watsonx-ai"

    apikey = os.getenv("WATSONX_APIKEY")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    if not (apikey and project_id):
        return "[watsonx.ai missing credentials] Set WATSONX_APIKEY and WATSONX_PROJECT_ID"

    try:
        creds = Credentials(url=url, api_key=apikey)
        gen_params = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 512,
            GenParams.TEMPERATURE: 0.2,
        }
        mdl = Model(model_id=model_id, params=gen_params, credentials=creds, project_id=project_id)
        resp = mdl.generate(prompt=prompt)
        # SDK returns dict with 'results'
        if isinstance(resp, dict):
            try:
                return resp["results"][0]["generated_text"]
            except Exception:
                return str(resp)
        return str(resp)
    except Exception as e:
        return f"[watsonx.ai error] {e}"


# 3) Hugging Face (local or API)
#    Option A: transformers local model path/id (offline ok if weights present)
#    Option B: text-generation-inference server / Inference API (requires HUGGINGFACE_TOKEN)

def call_hf_transformers(prompt: str, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens: int = 512) -> str:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception:
        return "[transformers missing] pip install transformers accelerate torch --extra-index-url https://download.pytorch.org/whl/cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.2)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"[HF transformers error] {e}"


# ------------------------------
# Finance helpers (toy calculations; for education only)
# ------------------------------

def project_savings(monthly_invest: float, annual_return_pct: float, years: int) -> float:
    r = annual_return_pct / 100 / 12
    n = years * 12
    if r == 0:
        return monthly_invest * n
    return monthly_invest * ((1 + r) ** n - 1) / r


def simple_tax_estimator(income: float, deductions: float = 0.0) -> Dict[str, float]:
    taxable = max(0.0, income - deductions)
    # Simplified progressive slabs (illustrative only, not legal advice)
    slabs = [
        (300000, 0.0),
        (300000, 0.05),
        (300000, 0.10),
        (300000, 0.15),
        (300000, 0.20),
        (math.inf, 0.30),
    ]
    tax = 0.0
    remaining = taxable
    for band, rate in slabs:
        portion = min(remaining, band)
        tax += portion * rate
        remaining -= portion
        if remaining <= 0:
            break
    cess = 0.04 * tax
    total = tax + cess
    return {"taxable": taxable, "tax": tax, "cess": cess, "total": total}


# ------------------------------
# Layout
# ------------------------------
_ensure_state()

with st.sidebar:
    st.markdown("""
    <div class="glass">
      <div class="badge">💸 <b>FinBuddy</b> · Personal Finance Copilot</div>
      <div class="small" style="margin-top:.5rem;">Choose your brain, tune safety, and chat about money like a pro.</div>
    </div>
    """, unsafe_allow_html=True)

    provider = st.selectbox(
        "Model Provider",
        ["Local Llama 3 (Ollama)", "IBM watsonx.ai (Granite)", "Hugging Face (transformers)"]
    )

    if provider == "Local Llama 3 (Ollama)":
        ollama_host = st.text_input("Ollama Host", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        ollama_model = st.text_input("Model", os.getenv("OLLAMA_MODEL", "llama3"))
    elif provider == "IBM watsonx.ai (Granite)":
        st.info("Set env: WATSONX_APIKEY, WATSONX_URL (optional), WATSONX_PROJECT_ID")
        watsonx_model = st.text_input("Granite Model ID", "ibm/granite-13b-instruct-v2")
    else:
        hf_model = st.text_input("HF Model (local or hub id)", "meta-llama/Meta-Llama-3-8B-Instruct")

    guardrails = st.toggle("Redact obvious PII from prompts", value=True)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)

    st.markdown("""
    <div class="glass small">
      ⚠️ Educational tool. Not investment, tax, or legal advice. Verify numbers before using.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="glass">
  <h1 style="margin:0">💸 FinBuddy</h1>
  <div class="small">Intelligent guidance for savings, taxes, and investments — with switchable LLM backends.</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='glass'><div class='kpi'>🪙</div><div class='kpi-sub'>Monthly SIP</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='glass'><div class='kpi'>📈</div><div class='kpi-sub'>Projected Corpus</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='glass'><div class='kpi'>🧾</div><div class='kpi-sub'>Tax Estimate</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='glass'><div class='kpi'>🛡️</div><div class='kpi-sub'>PII Guardrails</div></div>", unsafe_allow_html=True)

# ------------------------------
# Tabs
# ------------------------------
chat_tab, savings_tab, taxes_tab, invest_tab, whatif_tab = st.tabs([
    "💬 Coach", "💵 Savings", "🧾 Taxes", "📊 Investments", "🧪 What‑If"
])

# ---- Chat / Coach Tab ----
with chat_tab:
    st.subheader("Your Money Coach")
    user_input = st.text_area("Ask anything about savings, taxes, investments…", height=100, placeholder="e.g., Build a 10‑year plan combining SIP + emergency fund")
    send = st.button("Chat", type="primary")

    if send and user_input.strip():
        prompt = user_input.strip()
        if guardrails:
            prompt = _pii_redact(prompt)

        # Build a system prompt for finance‑aware coaching
        sys = (
            "You are a conservative personal finance coach."
            " Use plain language, show formulas where helpful, and separate advice into steps."
            " Include disclaimers when assumptions are made."
        )
        content = ""
        with st.spinner("Thinking…"):
            if provider == "Local Llama 3 (Ollama)":
                content = call_ollama(ollama_model, [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt}
                ], host=ollama_host, temperature=temperature)
            elif provider == "IBM watsonx.ai (Granite)":
                content = call_watsonx_granite(prompt=f"{sys}\nUser: {prompt}\nCoach:", model_id=watsonx_model)
            else:
                content = call_hf_transformers(prompt=f"<|system|>{sys}\n<|user|>{prompt}\n<|assistant|>", model_id=hf_model)

        st.session_state.chat.append(Msg("user", user_input))
        st.session_state.chat.append(Msg("assistant", content))

    # Render history
    for m in st.session_state.chat[-24:]:
        if m.role == "user":
            st.markdown(f"<div class='chat-bubble-user'>🧑‍💼 {m.content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-ai'>🤖 {m.content}</div>", unsafe_allow_html=True)

# ---- Savings Tab ----
with savings_tab:
    st.subheader("Savings Planner (SIP)")
    c1, c2, c3 = st.columns(3)
    with c1:
        monthly = st.number_input("Monthly Investment (₹)", 0.0, step=100.0, value=5000.0)
    with c2:
        rate = st.slider("Expected Annual Return %", 0.0, 24.0, 12.0, 0.5)
    with c3:
        years = st.slider("Years", 1, 40, 10, 1)

    if st.button("Project Corpus"):
        corpus = project_savings(monthly, rate, years)
        st.success(f"Projected corpus in {years} years: ₹{corpus:,.0f}")
        st.session_state.savings_plan = {"monthly": monthly, "rate": rate, "years": years, "corpus": corpus}

    if st.session_state.savings_plan:
        st.json(st.session_state.savings_plan)

# ---- Taxes Tab ----
with taxes_tab:
    st.subheader("Quick Tax Estimator (Illustrative)")
    i1, i2 = st.columns(2)
    with i1:
        income = st.number_input("Annual Gross Income (₹)", 0.0, step=5000.0, value=800000.0)
    with i2:
        deductions = st.number_input("Deductions (80C/80D/etc) (₹)", 0.0, step=5000.0, value=150000.0)

    if st.button("Estimate Tax"):
        est = simple_tax_estimator(income, deductions)
        st.info(f"Taxable Income: ₹{est['taxable']:,.0f}")
        st.warning(f"Income Tax: ₹{est['tax']:,.0f} · Health & Edu Cess (4%): ₹{est['cess']:,.0f}")
        st.success(f"Total Payable: ₹{est['total']:,.0f}")
        st.caption("This calculator is simplified for learning and may not match current laws.")

# ---- Investments Tab ----
with invest_tab:
    st.subheader("Portfolio Notes")
    st.markdown("""
    - Keep 6–12 months expenses as emergency fund (liquid/FD).  
    - Diversify: Equity index funds + Debt funds/FD/RD; optionally gold REIT/sovereign gold bonds.  
    - Rebalance yearly; align with risk tolerance and goals.  
    - Avoid high-cost products and unregulated promises.
    """)

    prompt = st.text_area("Ask the model to review a hypothetical allocation:",
                          "40% Nifty 50 index, 20% Nifty Midcap, 30% Short-term Debt, 10% Gold.")
    if st.button("Review Allocation"):
        sys = "Be a risk-aware investment analyst. Return pros/cons, risks, and rebalancing tips."
        with st.spinner("Analyzing allocation…"):
            if provider == "Local Llama 3 (Ollama)":
                out = call_ollama(ollama_model, [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt}
                ], host=ollama_host, temperature=temperature)
            elif provider == "IBM watsonx.ai (Granite)":
                out = call_watsonx_granite(prompt=f"{sys}\nUser: {prompt}\nAnalyst:", model_id=watsonx_model)
            else:
                out = call_hf_transformers(prompt=f"<|system|>{sys}\n<|user|>{prompt}\n<|assistant|>", model_id=hf_model)
        st.markdown(out)

# ---- What-If Tab ----
with whatif_tab:
    st.subheader("What‑If Simulator")
    c1, c2, c3 = st.columns(3)
    with c1:
        lump_sum = st.number_input("Lump Sum (₹)", 0.0, step=10000.0, value=100000.0)
    with c2:
        sip = st.number_input("Monthly SIP (₹)", 0.0, step=1000.0, value=5000.0)
    with c3:
        return_pct = st.slider("Annual Return %", 0.0, 24.0, 10.0, 0.5)

    horizon = st.slider("Years", 1, 40, 15)

    if st.button("Run Scenario"):
        # Future value of lump sum + SIP
        r = return_pct / 100 / 12
        n = horizon * 12
        fv_lump = lump_sum * ((1 + r) ** n)
        fv_sip = project_savings(sip, return_pct, horizon)
        total = fv_lump + fv_sip
        st.success(f"Future Value ≈ ₹{total:,.0f} (Lump: ₹{fv_lump:,.0f} + SIP: ₹{fv_sip:,.0f})")
        st.caption("Assumes constant returns; reality varies.")

# ------------------------------
# Footer / Toolbar
# ------------------------------
st.markdown("""
<div class="toolbar small">
  Tip: Switch providers in the sidebar to compare responses between Local Llama 3, IBM Granite on watsonx.ai, and Hugging Face models.
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Quickstart in-app docs
# ------------------------------
with st.expander("🛠️ Setup & Run (read me)"):
    st.markdown(
        """
        **1) Install**
        ```bash
        pip install streamlit requests ibm-watsonx-ai transformers accelerate torch --extra-index-url https://download.pytorch.org/whl/cpu
        ```

        **2) Local Llama 3 via Ollama**
        ```bash
        # Install Ollama from https://ollama.com
        ollama pull llama3
        export OLLAMA_HOST=http://localhost:11434
        streamlit run app.py
        ```

        **3) IBM watsonx.ai (Granite)**
        - Create a watsonx.ai project and obtain API key.  
        - Set environment variables:
        ```bash
        export WATSONX_APIKEY=your_key
        export WATSONX_URL=https://us-south.ml.cloud.ibm.com
        export WATSONX_PROJECT_ID=your_project_id
        ```
        - Choose provider **IBM watsonx.ai (Granite)** in the sidebar and use a model id like `ibm/granite-13b-instruct-v2`.

        **4) Hugging Face (Local or Hub)**
        - If you have the model locally (folder path) it will load offline.  
        - Or set `hf_model` to a hub id (internet needed on first load).

        **Notes**
        - This app is for education. Do **not** rely on it for compliance or filings.
        - The tax estimator uses simplified slabs and a 4% cess; check latest rules before use.
        - The PII redactor is minimal; do not paste sensitive info.
        """
    )
