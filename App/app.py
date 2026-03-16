"""
Udupi FIR RAG System — Streamlit App
=====================================
Run with:  streamlit run app.py
"""

import os
import re
import time
import random
import pickle
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import faiss
import streamlit as st
from rank_bm25 import BM25Okapi
import google.generativeai as genai

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title  = "Udupi FIR Assistant",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0d0d;
    color: #e8e8e8;
}

.main { background-color: #0d0d0d; }

.stApp { background-color: #0d0d0d; }

/* Header */
.fir-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #e94560;
    border-radius: 4px;
    padding: 24px 32px;
    margin-bottom: 24px;
}
.fir-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    color: #e94560;
    margin: 0;
    letter-spacing: 2px;
}
.fir-header p {
    color: #a0a0b0;
    margin: 6px 0 0 0;
    font-size: 0.9rem;
    letter-spacing: 1px;
}

/* Cards */
.fir-card {
    background: #141414;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.fir-card:hover { border-color: #e94560; }

.fir-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.fir-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 8px;
    border-radius: 2px;
    font-weight: 600;
    letter-spacing: 1px;
}
.badge-crime { background: #1a1a2e; color: #7b8cde; border: 1px solid #7b8cde; }
.badge-loc   { background: #1a2e1a; color: #56e94d; border: 1px solid #56e94d; }
.badge-score { background: #2e1a1a; color: #e94560; border: 1px solid #e94560; }

.fir-text {
    font-size: 0.85rem;
    color: #c0c0d0;
    line-height: 1.6;
    border-left: 2px solid #2a2a3e;
    padding-left: 12px;
    margin-top: 8px;
}

/* Answer box */
.answer-box {
    background: #0a1628;
    border: 1px solid #e94560;
    border-radius: 4px;
    padding: 20px 24px;
    margin-top: 16px;
}
.answer-box h4 {
    font-family: 'IBM Plex Mono', monospace;
    color: #e94560;
    font-size: 0.8rem;
    letter-spacing: 2px;
    margin: 0 0 12px 0;
}
.answer-text {
    color: #e8e8e8;
    font-size: 1rem;
    line-height: 1.8;
}

/* Metric boxes */
.metric-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
}
.metric-box {
    flex: 1;
    background: #141414;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 14px 16px;
    text-align: center;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #606070;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
}

/* Progress bar */
.progress-track {
    background: #2a2a3e;
    border-radius: 2px;
    height: 6px;
    margin-top: 6px;
}
.progress-fill {
    height: 6px;
    border-radius: 2px;
    transition: width 0.4s ease;
}

/* Diagnosis */
.diagnosis-box {
    border-radius: 4px;
    padding: 12px 16px;
    margin-top: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}
.diag-good    { background: #0a1e0a; border: 1px solid #56e94d; color: #56e94d; }
.diag-warn    { background: #1e1a0a; border: 1px solid #e9c44d; color: #e9c44d; }
.diag-bad     { background: #1e0a0a; border: 1px solid #e94560; color: #e94560; }
.diag-partial { background: #1a1a0a; border: 1px solid #a0a020; color: #c0c030; }

/* Sidebar */
.sidebar-section {
    background: #141414;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 14px;
    margin-bottom: 14px;
}
.sidebar-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #e94560;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

/* Comparison table */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 8px;
}
.compare-table th {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #606070;
    letter-spacing: 1px;
    border-bottom: 1px solid #2a2a3e;
    padding: 6px 10px;
    text-align: left;
}
.compare-table td {
    padding: 8px 10px;
    border-bottom: 1px solid #1a1a2e;
    color: #c0c0d0;
    vertical-align: top;
}
.compare-table tr:hover td { background: #1a1a2e; }

/* Input */
.stTextInput input {
    background: #141414 !important;
    border: 1px solid #2a2a3e !important;
    color: #e8e8e8 !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.stTextInput input:focus {
    border-color: #e94560 !important;
    box-shadow: 0 0 0 1px #e94560 !important;
}

/* Button */
.stButton button {
    background: #e94560 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 1px !important;
    font-size: 0.8rem !important;
    padding: 10px 24px !important;
}
.stButton button:hover {
    background: #c73550 !important;
}

/* Selectbox */
.stSelectbox div[data-baseweb="select"] > div {
    background: #141414 !important;
    border: 1px solid #2a2a3e !important;
    color: #e8e8e8 !important;
}

/* Slider */
.stSlider .stSlider > div { color: #e8e8e8; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #141414;
    border-bottom: 1px solid #2a2a3e;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
    color: #606070;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    color: #e94560 !important;
    border-bottom: 2px solid #e94560 !important;
    background: transparent !important;
}

/* Spinner */
.stSpinner { color: #e94560 !important; }

hr { border-color: #2a2a3e; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class SimpleEmbedderV4(nn.Module):
    def __init__(self, vocab_size, dim=64):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, dim)
        self.attention = nn.Linear(dim, 1)
        self.network   = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, dim)
        )

    def forward(self, x):
        embedded = self.embed(x)
        scores   = self.attention(embedded)
        weights  = torch.softmax(scores, dim=0)
        pooled   = (weights * embedded).sum(dim=0)
        return self.network(pooled)


# ─────────────────────────────────────────────
# TEXT PROCESSING
# ─────────────────────────────────────────────
def normalize_text(text):
    text = str(text)
    text = re.sub(r'[^\u0C80-\u0CFF\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text):
    return normalize_text(text).split()

def encode_doc(tokens, word2idx):
    if isinstance(tokens, str):
        tokens = tokenize(tokens)
    return [word2idx[t] for t in tokens if t in word2idx]


# ─────────────────────────────────────────────
# LOAD ALL RESOURCES (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_everything(data_path, model_path, gemini_key=""):
    status = {}

    # 1 — Load data
    df = pd.read_csv(data_path, header=0).dropna()
    df.reset_index(inplace=True)
    texts = df['Crime Description'].apply(normalize_text).tolist()
    status['records'] = len(df)

    # 2 — Build vocab + crime groups
    crime_groups  = defaultdict(list)
    vocab_counter = Counter()
    for idx, row in df.iterrows():
        tokens = tokenize(texts[idx])
        vocab_counter.update(tokens)
        crime_groups[row['Crime Type']].append(tokens)

    word2idx = {w: i for i, w in enumerate(vocab_counter.keys())}
    idx2word = {i: w for w, i in word2idx.items()}
    status['vocab'] = len(word2idx)
    status['crime_types'] = len(crime_groups)

    # 3 — Load trained model
    device     = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model      = SimpleEmbedderV4(checkpoint['vocab_size'],
                                   dim=checkpoint['embed_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    word2idx = checkpoint['word2idx']
    status['model'] = 'loaded'

    # 4 — Generate embeddings + FAISS index
    embed_dim          = checkpoint['embed_dim']
    X_customembeddings = np.empty((len(df), embed_dim))
    with torch.no_grad():
        for i in range(len(df)):
            tokens  = tokenize(texts[i])
            encoded = encode_doc(tokens, word2idx)
            if len(encoded) == 0:
                X_customembeddings[i] = np.zeros(embed_dim)
            else:
                X_customembeddings[i] = model(
                    torch.tensor(encoded)
                ).numpy()

    custom_index = faiss.IndexFlatL2(embed_dim)
    custom_index.add(X_customembeddings.astype('float32'))
    status['faiss'] = custom_index.ntotal

    # 5 — BM25 index
    tokenized_corpus = [tokenize(t) for t in texts]
    bm25             = BM25Okapi(tokenized_corpus)
    status['bm25'] = 'built'

    # 6 — Gemini (optional)
    gemini = None
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            gemini = genai.GenerativeModel(
                model_name        = 'gemini-2.0-flash',
                generation_config = genai.GenerationConfig(
                    temperature       = 0.15,
                    top_p             = 0.9,
                    max_output_tokens = 2048,
                )
            )
            status['gemini'] = 'connected'
        except Exception as e:
            status['gemini'] = f'failed: {e}'
    else:
        status['gemini'] = 'not configured'

    return {
        'df'           : df,
        'texts'        : texts,
        'model'        : model,
        'word2idx'     : word2idx,
        'crime_groups' : crime_groups,
        'custom_index' : custom_index,
        'bm25'         : bm25,
        'gemini'       : gemini,
        'embed_dim'    : embed_dim,
        'status'       : status,
    }


# ─────────────────────────────────────────────
# SEARCH FUNCTIONS
# ─────────────────────────────────────────────
def hybrid_search(query, R, k=5, alpha=0.5):
    df           = R['df']
    model        = R['model']
    word2idx     = R['word2idx']
    custom_index = R['custom_index']
    bm25         = R['bm25']
    texts        = R['texts']

    query_tokens = tokenize(query)
    bm25_scores  = bm25.get_scores(query_tokens)

    encoded = encode_doc(query_tokens, word2idx)
    if len(encoded) == 0:
        top_k = bm25_scores.argsort()[-k:][::-1]
        return df.iloc[top_k], bm25_scores[top_k], list(top_k)

    model.eval()
    with torch.no_grad():
        q_emb = model(
            torch.tensor(encoded)
        ).numpy().reshape(1, -1).astype('float32')

    distances, indices = custom_index.search(q_emb, len(texts))

    emb_scores             = np.zeros(len(texts))
    emb_scores[indices[0]] = 1 / (1 + distances[0])

    def normalize(s):
        mn, mx = s.min(), s.max()
        return s if mx - mn < 1e-9 else (s - mn) / (mx - mn)

    combined = alpha * normalize(bm25_scores) + (1 - alpha) * normalize(emb_scores)
    top_k    = combined.argsort()[-k:][::-1]

    return df.iloc[top_k], combined[top_k], list(top_k)


def rag_generate(query, context_firs, gemini):
    context = "\n\n---\n\n".join(context_firs)
    prompt  = f"""ನೀವು ಕರ್ನಾಟಕ ಪೊಲೀಸ್ ಇಲಾಖೆಗೆ ಸಹಾಯ ಮಾಡುವ AI ಸಹಾಯಕ.
ನೀವು ಕೇವಲ FIR ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ ಉತ್ತರಿಸುತ್ತೀರಿ.

ಕಟ್ಟುನಿಟ್ಟಾದ ನಿಯಮಗಳು:
- ಕೆಳಗಿನ FIR ವಿವರಗಳಲ್ಲಿ ಇರುವ ಮಾಹಿತಿಯನ್ನು ಮಾತ್ರ ಬಳಸಿ
- FIR ವಿವರಗಳಲ್ಲಿ ಇಲ್ಲದ ಯಾವುದೇ ಮಾಹಿತಿಯನ್ನು ಸೇರಿಸಬಾರದು
- ಇಂಗ್ಲಿಷ್ ಬಳಸಬಾರದು
- ಉತ್ತರ ಕನ್ನಡದಲ್ಲಿರಬೇಕು
- ಯಾವುದೇ ಪಟ್ಟಿ, ಅಂಕಿಗಳು, ಚಿಹ್ನೆಗಳು ಬಳಸಬಾರದು
- ಒಂದೇ ಪ್ಯಾರಾಗ್ರಾಫ್‌ನಲ್ಲಿ ನೇರವಾಗಿ ಉತ್ತರಿಸಿ
- FIR ನಲ್ಲಿರುವ ವ್ಯಕ್ತಿ, ಸ್ಥಳ, ದಿನಾಂಕ, ಘಟನೆ ಮಾತ್ರ ಹೇಳಿ

FIR ವಿವರಗಳು:
{context}

ಪ್ರಶ್ನೆ: {query}

ಉತ್ತರ (ಒಂದೇ ಪ್ಯಾರಾಗ್ರಾಫ್, ಕನ್ನಡದಲ್ಲಿ):"""

    try:
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f'⚠️ Generation error: {e}'


def faithfulness_check(answer, context_firs):
    answer_tokens  = set(tokenize(answer))
    context_tokens = set()
    for fir in context_firs:
        context_tokens.update(tokenize(fir))
    if len(answer_tokens) == 0:
        return 0.0
    return round(len(answer_tokens & context_tokens) / len(answer_tokens), 4)


def context_relevance_score(query, retrieved_df, crime_groups):
    query_tokens = set(tokenize(query))
    best_type, best_score = None, 0
    for crime_type in crime_groups:
        overlap = len(query_tokens & set(tokenize(crime_type)))
        if overlap > best_score:
            best_score = overlap
            best_type  = crime_type
    if best_type is None:
        return 0.0, 'Unknown'
    matches = sum(1 for t in retrieved_df['Crime Type'] if t == best_type)
    score   = matches / len(retrieved_df) if len(retrieved_df) > 0 else 0.0
    return round(score, 4), best_type


def get_diagnosis(cr, faith, hit):
    if cr >= 0.6 and faith >= 0.6:
        return 'good', '✅ Good — retrieval + answer both grounded'
    elif cr >= 0.6 and faith < 0.3:
        return 'warn', '⚠️ Retrieval ok but answer not grounded enough'
    elif cr == 0.0 and not hit:
        return 'bad',  '❌ Retrieval failed — wrong crime type fetched'
    elif hit and faith < 0.3:
        return 'warn', '⚠️ Hit found but answer uses few FIR words'
    else:
        return 'partial', '🔶 Partial — try increasing k or tuning alpha'


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">⚙ CONFIGURATION</div>
    </div>
    """, unsafe_allow_html=True)

    data_path  = st.text_input("CSV Path",   value="UdupiCrimeData.csv")
    model_path = st.text_input("Model Path", value="fir_embedder_v4.pth")
    gemini_key = st.text_input("Gemini API Key", type="password",
                                placeholder="AIzaSy...")

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">🔧 SEARCH SETTINGS</div>
    </div>
    """, unsafe_allow_html=True)

    alpha = st.slider("Alpha (BM25 ←→ Embedding)",
                      0.0, 1.0, 0.5, 0.1,
                      help="1.0 = pure BM25 | 0.0 = pure embedding")
    k     = st.slider("Results to Retrieve (k)", 1, 10, 3)

    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                color:#606070; margin-top:8px;">
        BM25 weight   : {alpha:.1f}<br>
        Embed weight  : {1-alpha:.1f}<br>
        Top-k         : {k}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">🤖 GENERATION</div>
    </div>
    """, unsafe_allow_html=True)

    enable_generation = st.toggle(
        "Enable Gemini Generation",
        value=False,
        help="OFF = show retrieved FIRs only (no API call)\nON = retrieved FIRs + Gemini answer"
    )

    if enable_generation:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    color:#56e94d; margin-top:4px;">
            ✅ Generation ON — Gemini will be called
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    color:#e94560; margin-top:4px;">
            ⏸ Generation OFF — retrieval only mode
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    load_btn = st.button("⚡ LOAD SYSTEM", use_container_width=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="fir-header">
    <h1>🔍 UDUPI FIR ASSISTANT</h1>
    <p>Retrieval Augmented Generation · Kannada FIR Dataset · Custom Embeddings + BM25</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD SYSTEM
# ─────────────────────────────────────────────
if 'R' not in st.session_state:
    st.session_state.R = None

if load_btn:
    if not os.path.exists(data_path):
        st.error(f"CSV not found: `{data_path}`")
    elif not os.path.exists(model_path):
        st.error(f"Model not found: `{model_path}`")
    else:
        with st.spinner("Loading data, model, FAISS index, BM25..."):
            try:
                R = load_everything(data_path, model_path, gemini_key)
                st.session_state.R = R
                s = R['status']
                gemini_status = f"· Gemini {s['gemini']}" if gemini_key else "· Gemini not configured (retrieval only)"
                st.success(
                    f"✅ Loaded — {s['records']} records · "
                    f"{s['vocab']} vocab · "
                    f"{s['crime_types']} crime types · "
                    f"FAISS {s['faiss']} vectors {gemini_status}"
                )
            except Exception as e:
                st.error(f"Load failed: {e}")


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
if st.session_state.R is None:
    st.markdown("""
    <div style="text-align:center; padding:80px 0; color:#404050;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:3rem;">⬆</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem;
                    letter-spacing:2px; margin-top:12px;">
            CONFIGURE PATHS + API KEY IN SIDEBAR<br>THEN CLICK LOAD SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

R = st.session_state.R

tab1, tab2, tab3 = st.tabs([
    "  🔍  QUERY + ANSWER  ",
    "  ⚖️  COMPARE METHODS  ",
    "  📊  METRICS DASHBOARD  ",
])


# ─────────────────────────────────────────────
# TAB 1 — QUERY + ANSWER
# ─────────────────────────────────────────────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your query in Kannada",
            placeholder="ಮೊಬೈಲ್ ಫೋನ್ ಕಳವು",
            label_visibility="collapsed"
        )
    with col2:
        search_btn = st.button("SEARCH →", use_container_width=True)

    # sample queries
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                color:#404050; margin-top:6px; letter-spacing:1px;">
        EXAMPLES → ಮೊಬೈಲ್ ಫೋನ್ ಕಳವು &nbsp;|&nbsp; ಮನೆಯಲ್ಲಿ ಕಳ್ಳತನ
        &nbsp;|&nbsp; ರಸ್ತೆ ಅಪಘಾತ &nbsp;|&nbsp; ಗಾಂಜಾ ಸೇವನೆ
        &nbsp;|&nbsp; ಅಸ್ವಾಭಾವಿಕ ಮರಣ
    </div>
    """, unsafe_allow_html=True)

    if search_btn and query.strip():
        with st.spinner("Retrieving FIRs..."):
            retrieved_df, scores, indices = hybrid_search(
                query, R, k=k, alpha=alpha
            )
            context_firs = list(retrieved_df['Crime Description'])

        # ── Retrieved FIRs ────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                    color:#e94560; letter-spacing:2px; margin-bottom:12px;">
            📄 RETRIEVED FIRs &nbsp;·&nbsp;
            <span style="color:#606070;">
                alpha={alpha:.1f} · k={k} · {len(retrieved_df)} results
            </span>
        </div>
        """, unsafe_allow_html=True)

        for i, (_, row) in enumerate(retrieved_df.iterrows()):
            crime_type = row['Crime Type']
            location   = row['Location']
            score      = scores[i]
            desc       = row['Crime Description']

            # show full description in expander
            st.markdown(f"""
            <div class="fir-card">
                <div class="fir-card-header">
                    <div style="display:flex; gap:8px; flex-wrap:wrap;">
                        <span class="fir-badge badge-score">#{i+1} · {score:.3f}</span>
                        <span class="fir-badge badge-crime">{crime_type}</span>
                        <span class="fir-badge badge-loc">{location}</span>
                    </div>
                </div>
                <div class="fir-text">{desc[:400]}...</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"View full FIR #{i+1}"):
                st.markdown(f"""
                <div style="font-size:0.85rem; color:#c0c0d0;
                            line-height:1.8; font-family:'IBM Plex Sans',sans-serif;">
                    {desc}
                </div>
                """, unsafe_allow_html=True)

        # ── Generation (toggle controlled) ───────────────
        st.markdown("<br>", unsafe_allow_html=True)

        if not enable_generation:
            st.markdown("""
            <div style="background:#141414; border:1px dashed #2a2a3e;
                        border-radius:4px; padding:16px 20px;
                        font-family:'IBM Plex Mono',monospace;
                        font-size:0.75rem; color:#404050;
                        text-align:center; letter-spacing:1px;">
                ⏸ GENERATION PAUSED — toggle ON in sidebar to enable Gemini answer
            </div>
            """, unsafe_allow_html=True)

        else:
            # check API key
            if not R['gemini']:
                st.warning("⚠️ Gemini not configured — add API key in sidebar and reload.")
            else:
                with st.spinner("Generating answer with Gemini..."):
                    answer = rag_generate(query, context_firs, R['gemini'])

                st.markdown(f"""
                <div class="answer-box">
                    <h4>🤖 GEMINI ANSWER</h4>
                    <div class="answer-text">{answer}</div>
                </div>
                """, unsafe_allow_html=True)

                # metrics
                faith              = faithfulness_check(answer, context_firs)
                cr_score, inf_type = context_relevance_score(
                    query, retrieved_df, R['crime_groups']
                )
                hit                = inf_type in list(retrieved_df['Crime Type'])
                diag_cls, diag_msg = get_diagnosis(cr_score, faith, hit)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                            color:#e94560; letter-spacing:2px; margin-bottom:12px;">
                    📊 METRICS
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">CONTEXT RELEVANCE</div>
                        <div class="metric-value" style="color:#7b8cde;">{cr_score:.2f}</div>
                        <div class="progress-track">
                            <div class="progress-fill"
                                 style="width:{cr_score*100:.0f}%;background:#7b8cde;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with m2:
                    hit_color = '#56e94d' if hit else '#e94560'
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">HIT RATE</div>
                        <div class="metric-value" style="color:{hit_color};">
                            {"1.00" if hit else "0.00"}
                        </div>
                        <div style="font-size:0.7rem;color:{hit_color};margin-top:4px;">
                            {"✅ Found" if hit else "❌ Not Found"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with m3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">FAITHFULNESS</div>
                        <div class="metric-value" style="color:#e9c44d;">{faith:.2f}</div>
                        <div class="progress-track">
                            <div class="progress-fill"
                                 style="width:{faith*100:.0f}%;background:#e9c44d;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with m4:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">INFERRED TYPE</div>
                        <div style="font-family:'IBM Plex Mono',monospace;
                                    font-size:0.8rem; color:#a0a0b0;
                                    margin-top:6px; line-height:1.4;">
                            {inf_type if inf_type else 'Unknown'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                diag_map = {
                    'good':'diag-good','warn':'diag-warn',
                    'bad':'diag-bad','partial':'diag-partial'
                }
                st.markdown(f"""
                <div class="diagnosis-box {diag_map[diag_cls]}">
                    🔧 DIAGNOSIS &nbsp;→&nbsp; {diag_msg}
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 2 — COMPARE METHODS
# ─────────────────────────────────────────────
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        cmp_query = st.text_input(
            "Query for comparison",
            placeholder="ಮೊಬೈಲ್ ಫೋನ್ ಕಳವು",
            label_visibility="collapsed",
            key="cmp_query"
        )
    with col2:
        cmp_btn = st.button("COMPARE →", use_container_width=True)

    if cmp_btn and cmp_query.strip():
        methods = [
            ("BM25 Only",      1.0),
            ("Hybrid (50/50)", 0.5),
            ("Embedding Only", 0.0),
        ]

        cols = st.columns(3)

        for col, (method_name, a) in zip(cols, methods):
            with col:
                st.markdown(f"""
                <div style="font-family:'IBM Plex Mono',monospace;
                            font-size:0.7rem; color:#e94560;
                            letter-spacing:2px; margin-bottom:12px;
                            border-bottom:1px solid #2a2a3e;
                            padding-bottom:8px;">
                    {method_name}
                </div>
                """, unsafe_allow_html=True)

                with st.spinner(f"Running {method_name}..."):
                    ret_df, ret_scores, ret_idx = hybrid_search(
                        cmp_query, R, k=k, alpha=a
                    )

                for i, (_, row) in enumerate(ret_df.iterrows()):
                    score = ret_scores[i]
                    st.markdown(f"""
                    <div class="fir-card">
                        <div style="display:flex;gap:6px;margin-bottom:8px;
                                    flex-wrap:wrap;">
                            <span class="fir-badge badge-score">
                                #{i+1} · {score:.3f}
                            </span>
                            <span class="fir-badge badge-crime">
                                {row['Crime Type'][:20]}
                            </span>
                            <span class="fir-badge badge-loc">
                                {row['Location'][:12]}
                            </span>
                        </div>
                        <div class="fir-text">
                            {row['Crime Description'][:200]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 3 — METRICS DASHBOARD
# ─────────────────────────────────────────────
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                color:#e94560; letter-spacing:2px; margin-bottom:16px;">
        📊 BATCH EVALUATION — runs all queries, computes all metrics
    </div>
    """, unsafe_allow_html=True)

    default_queries = (
        "ಮೊಬೈಲ್ ಫೋನ್ ಕಳವು\n"
        "ಮನೆಯಲ್ಲಿ ಕಳ್ಳತನ\n"
        "ರಸ್ತೆ ಅಪಘಾತ\n"
        "ಗಾಂಜಾ ಸೇವನೆ\n"
        "ಅಸ್ವಾಭಾವಿಕ ಮರಣ"
    )

    batch_input = st.text_area(
        "Queries (one per line)",
        value=default_queries,
        height=150
    )
    delay  = st.slider("Delay between Gemini calls (sec)", 3, 15, 5)
    run_btn = st.button("▶ RUN BATCH EVALUATION", use_container_width=True)

    if run_btn:
        queries = [q.strip() for q in batch_input.strip().split('\n') if q.strip()]
        results = []

        progress = st.progress(0)
        status   = st.empty()

        for i, q in enumerate(queries):
            status.markdown(
                f"<span style='font-family:IBM Plex Mono;font-size:0.8rem;"
                f"color:#a0a0b0;'>Processing {i+1}/{len(queries)}: {q}</span>",
                unsafe_allow_html=True
            )

            if i > 0:
                time.sleep(delay)

            try:
                ret_df, scores, indices = hybrid_search(q, R, k=k, alpha=alpha)
                context_firs = list(ret_df['Crime Description'])
                answer       = rag_generate(q, context_firs, R['gemini'])

                faith              = faithfulness_check(answer, context_firs)
                cr_score, inf_type = context_relevance_score(
                    q, ret_df, R['crime_groups']
                )
                hit                = inf_type in list(ret_df['Crime Type'])
                diag_cls, diag_msg = get_diagnosis(cr_score, faith, hit)

                results.append({
                    'Query'            : q,
                    'Inferred Type'    : inf_type,
                    'Context Relevance': cr_score,
                    'Hit Rate'         : '✅' if hit else '❌',
                    'Faithfulness'     : faith,
                    'Diagnosis'        : diag_msg,
                    'Answer'           : answer,
                })
            except Exception as e:
                results.append({
                    'Query'            : q,
                    'Inferred Type'    : 'Error',
                    'Context Relevance': 0.0,
                    'Hit Rate'         : '❌',
                    'Faithfulness'     : 0.0,
                    'Diagnosis'        : f'Error: {e}',
                    'Answer'           : '',
                })

            progress.progress((i + 1) / len(queries))

        status.empty()
        progress.empty()

        # summary stats
        valid = [r for r in results if r['Inferred Type'] != 'Error']
        if valid:
            avg_cr    = sum(r['Context Relevance'] for r in valid) / len(valid)
            avg_faith = sum(r['Faithfulness']       for r in valid) / len(valid)
            hit_count = sum(1 for r in valid if r['Hit Rate'] == '✅')

            c1, c2, c3, c4 = st.columns(4)
            for col, label, val, color in [
                (c1, "QUERIES RUN",        str(len(valid)),        "#7b8cde"),
                (c2, "AVG CONTEXT REL",    f"{avg_cr:.2f}",        "#7b8cde"),
                (c3, "HIT RATE",           f"{hit_count}/{len(valid)}", "#56e94d"),
                (c4, "AVG FAITHFULNESS",   f"{avg_faith:.2f}",     "#e9c44d"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color:{color};">{val}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # results table
        for r in results:
            diag_cls, _ = get_diagnosis(
                r['Context Relevance'],
                r['Faithfulness'],
                r['Hit Rate'] == '✅'
            )
            diag_map = {'good':'diag-good','warn':'diag-warn',
                        'bad':'diag-bad','partial':'diag-partial'}

            with st.expander(f"  {r['Query']}  ·  CR={r['Context Relevance']:.2f}  ·  Faith={r['Faithfulness']:.2f}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">CONTEXT RELEVANCE</div>
                        <div class="metric-value" style="color:#7b8cde;">
                            {r['Context Relevance']:.2f}
                        </div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">HIT RATE</div>
                        <div class="metric-value"
                             style="color:{'#56e94d' if r['Hit Rate']=='✅' else '#e94560'};">
                            {r['Hit Rate']}
                        </div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">FAITHFULNESS</div>
                        <div class="metric-value" style="color:#e9c44d;">
                            {r['Faithfulness']:.2f}
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="answer-box" style="margin-top:12px;">
                    <h4>🤖 ANSWER</h4>
                    <div class="answer-text">{r['Answer']}</div>
                </div>
                <div class="diagnosis-box {diag_map[diag_cls]}" style="margin-top:10px;">
                    {r['Diagnosis']}
                </div>
                """, unsafe_allow_html=True)

        # download results
        result_df = pd.DataFrame([{
            'Query'            : r['Query'],
            'Context Relevance': r['Context Relevance'],
            'Hit Rate'         : r['Hit Rate'],
            'Faithfulness'     : r['Faithfulness'],
            'Diagnosis'        : r['Diagnosis'],
            'Answer'           : r['Answer'],
        } for r in results])

        st.download_button(
            label     = "⬇ DOWNLOAD RESULTS CSV",
            data      = result_df.to_csv(index=False),
            file_name = "fir_eval_results.csv",
            mime      = "text/csv",
        )
