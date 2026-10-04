"""
Vectorless RAG with Google Gemini API & Streamlit.
Zero embeddings. Zero vector databases. Pure lexical BM25 + Entity Knowledge Graph + Gemini reasoning.
"""

import os
import sys
import time
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Ensure local modules are accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from rag_engine import VectorlessRAGEngine, AVAILABLE_MODELS
from sample_data import SAMPLE_DOCS

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Vectorless RAG | Gemini & BM25",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Rich Aesthetics)
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #93c5fd, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.5);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #60a5fa;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }

    /* Citation Cards */
    .citation-card {
        background: rgba(15, 23, 42, 0.7);
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 6px;
    }
    .token-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin-right: 4px;
        margin-bottom: 4px;
    }

    /* Fast badge */
    .speed-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "engine" not in st.session_state:
    st.session_state.engine = VectorlessRAGEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "auto_loaded_samples" not in st.session_state:
    # Auto-load benchmark sample documents on first run
    for doc_name, content in SAMPLE_DOCS.items():
        st.session_state.engine.ingest_text(content, doc_name)
    st.session_state.auto_loaded_samples = True

engine: VectorlessRAGEngine = st.session_state.engine

# ==================== SIDEBAR CONFIGURATION ====================
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    
    # API Key Input
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=engine.api_key or env_key,
        type="password",
        help="Reads from .env automatically or paste your key here."
    )
    if api_key_input != engine.api_key:
        engine.set_api_key(api_key_input)
        if api_key_input:
            st.success("API Key updated!")

    # Model Selection
    selected_model = st.selectbox(
        "Gemini Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(engine.model_name) if engine.model_name in AVAILABLE_MODELS else 0,
        help="Select the Gemini model for grounded synthesis."
    )
    if selected_model != engine.model_name:
        engine.set_model(selected_model)

    st.divider()

    st.markdown("### 🔬 Retrieval Algorithm")
    retrieval_mode = st.radio(
        "Search Strategy",
        options=["BM25", "Graph-Boosted BM25", "Hierarchical Section"],
        index=1,
        help=(
            "• BM25: Pure probabilistic term matching with IDF weighting\n"
            "• Graph-Boosted: BM25 + Co-occurrence Entity Graph bonus\n"
            "• Hierarchical Section: Preserves document header context"
        )
    )

    top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=10, value=4)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    with st.expander("🛠️ Advanced BM25 Tuning"):
        bm25_k1 = st.slider("k1 (Term Frequency Saturation)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        bm25_b = st.slider("b (Document Length Penalty)", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
        if bm25_k1 != engine.indexer.k1 or bm25_b != engine.indexer.b:
            engine.indexer.k1 = bm25_k1
            engine.indexer.b = bm25_b
            if engine.indexer.chunks:
                engine.indexer.build(engine.indexer.chunks)
            st.caption("BM25 parameters re-calibrated.")

    st.divider()

    st.markdown("### 📚 Knowledge Base Actions")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🔄 Reset Samples", use_container_width=True):
            engine.clear_documents()
            for doc_name, content in SAMPLE_DOCS.items():
                engine.ingest_text(content, doc_name)
            st.rerun()
    with col_sb2:
        if st.button("🗑️ Clear All", use_container_width=True):
            engine.clear_documents()
            st.session_state.chat_history = []
            st.rerun()

    st.caption("⚡ **Vectorless RAG Advantage:** No embeddings, no vector DB downtime, zero floating-point distance errors, exact acronym matching.")


# ==================== HERO SECTION & STATS ====================
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ Vectorless Architecture • Gemini 2026 Ready</div>
    <div class="hero-title">Vectorless RAG with Gemini API</div>
    <div class="hero-subtitle">
        High-precision retrieval augmented generation powered by BM25Okapi/BM25Plus probabilistic indexing, 
        Entity-Co-occurrence Knowledge Graphs, and Google Gemini grounding.
    </div>
</div>
""", unsafe_allow_html=True)

# Top Metrics Row
num_docs = len(set(c.doc_name for c in engine.indexer.chunks))
num_chunks = len(engine.indexer.chunks)
vocab_size = len(engine.indexer.doc_freqs)
graph_stats = engine.indexer.get_graph_summary()

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{num_docs}</div>
        <div class="metric-label">Documents</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{num_chunks}</div>
        <div class="metric-label">Granular Chunks</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{vocab_size}</div>
        <div class="metric-label">Inverted Vocab</div>
    </div>
    """, unsafe_allow_html=True)
with col_m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{graph_stats['num_nodes']}</div>
        <div class="metric-label">Entity Nodes</div>
    </div>
    """, unsafe_allow_html=True)
with col_m5:
    api_status = "Connected 🟢" if engine.client else "Missing Key 🔴"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size:1.15rem; margin-top:8px;">{selected_model}</div>
        <div class="metric-label">{api_status}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==================== MAIN TABS ====================
tab_chat, tab_search, tab_index, tab_docs = st.tabs([
    "💬 Grounded Chat & Q&A",
    "🔍 Vectorless Search Inspector",
    "📊 Inverted Index & Graph",
    "📁 Document Manager"
])


# ==================== TAB 1: CHAT & GROUNDED Q&A ====================
with tab_chat:
    st.markdown("##### 💡 Suggested Questions (Test Exact Lexical Retrieval):")
    q_col1, q_col2, q_col3 = st.columns(3)
    
    preset_q = None
    if q_col1.button("💰 NovaCorp Revenue & Gross Margin", use_container_width=True):
        preset_q = "What was NovaCorp's total revenue, gross margin, and ARR in Q3 2025?"
    if q_col2.button("⚖️ EU AI Act Article 71 Fines", use_container_width=True):
        preset_q = "What are the administrative fines under Article 71 of the EU AI Act for prohibited AI practices?"
    if q_col3.button("⚛️ Shor's Algorithm Qubit Requirements", use_container_width=True):
        preset_q = "According to the Helios-X9 benchmarks, how many logical and physical qubits are required for Shor's algorithm to factor 2048-bit RSA?"

    # Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg and msg["metadata"]:
                meta = msg["metadata"]
                with st.expander(f"📑 Sources & BM25 Scoring ({len(meta.get('chunks', []))} chunks | Retrieval: {meta.get('retrieval_time', 0)}s)"):
                    for item in meta.get("chunks", []):
                        chunk = item["chunk"]
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span>{chunk.source_tag}</span>
                                <span class="speed-badge">BM25: {item['score']}</span>
                            </div>
                            <div style="font-size: 0.9rem; color: #e2e8f0; margin-bottom: 8px;">
                                {chunk.content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if item.get("matched_terms"):
                            st.caption("Matched Terms & Postings Contribution:")
                            for term, stats in item["matched_terms"].items():
                                st.markdown(f"`{term}` (TF: {stats['tf']} | DF: {stats['df']} | IDF: {stats['idf']})", help="BM25 Token Contribution")

    # Handle Input
    user_input = st.chat_input("Ask a question based on your indexed documents...")
    final_query = preset_q if preset_q else user_input

    if final_query:
        # Append User Message
        st.session_state.chat_history.append({"role": "user", "content": final_query})
        with st.chat_message("user"):
            st.markdown(final_query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            retrieval_meta = {}

            # Execute streaming generation
            with st.spinner("⚡ Performing Vectorless BM25 retrieval & Gemini synthesis..."):
                gen_stream = engine.generate_answer_stream(
                    query=final_query,
                    retrieval_mode=retrieval_mode,
                    top_k=top_k,
                    temperature=temperature
                )
                
                for packet in gen_stream:
                    if packet["type"] == "metadata":
                        retrieval_meta = {
                            "chunks": packet["retrieved_chunks"],
                            "retrieval_time": packet["retrieval_time"]
                        }
                    elif packet["type"] == "token":
                        full_response += packet["text"]
                        response_container.markdown(full_response + "▌")
                    elif packet["type"] == "done":
                        retrieval_meta["total_time"] = packet["total_time"]

            response_container.markdown(full_response)

            # Show inline citations preview
            if retrieval_meta.get("chunks"):
                with st.expander(f"📑 Sources & BM25 Scoring ({len(retrieval_meta['chunks'])} chunks | Retrieval: {retrieval_meta['retrieval_time']}s)"):
                    for item in retrieval_meta["chunks"]:
                        chunk = item["chunk"]
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span>{chunk.source_tag}</span>
                                <span class="speed-badge">Score: {item['score']}</span>
                            </div>
                            <div style="font-size: 0.9rem; color: #e2e8f0; margin-bottom: 8px;">
                                {chunk.content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if item.get("matched_terms"):
                            badges = " ".join([f'<span class="token-badge">{t} (TF:{d["tf"]}|IDF:{d["idf"]})</span>' for t, d in item["matched_terms"].items()])
                            st.markdown(f"**Term Matches:** {badges}", unsafe_allow_html=True)

            # Store in history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "metadata": retrieval_meta
            })


# ==================== TAB 2: SEARCH & BM25 INSPECTOR ====================
with tab_search:
    st.markdown("### 🔍 Vectorless Search & BM25 Diagnostic Lab")
    st.caption("Inspect exact term frequencies, inverse document frequencies, and graph traversal boosts without invoking the LLM.")

    search_query = st.text_input("Enter Diagnostic Search Query", value="Shor's Algorithm 2048-bit RSA physical qubits")
    
    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        bm25_candidates = engine.retrieve(search_query, mode="BM25", top_k=top_k)
        st.markdown(f"#### 🎯 Pure BM25 Results ({len(bm25_candidates)})")
        for res in bm25_candidates:
            c = res["chunk"]
            with st.expander(f"Rank #{res['rank']} • Score: {res['score']} • {c.source_tag}", expanded=True):
                st.write(c.content)
                if res.get("matched_terms"):
                    st.caption("Matched Query Tokens:")
                    st.json(res["matched_terms"])

    with col_s2:
        graph_candidates = engine.retrieve(search_query, mode="Graph-Boosted BM25", top_k=top_k)
        st.markdown(f"#### 🕸️ Graph-Boosted BM25 Results ({len(graph_candidates)})")
        for res in graph_candidates:
            c = res["chunk"]
            boost_txt = f"+{res['graph_boost']} Graph Bonus" if res.get('graph_boost', 0) > 0 else "No boost"
            with st.expander(f"Rank #{res['rank']} • Total: {res['score']} ({boost_txt}) • {c.source_tag}", expanded=True):
                st.write(c.content)
                if res.get("graph_links"):
                    st.info(f"🔗 Traversed Graph Links: {', '.join(res['graph_links'])}")


# ==================== TAB 3: INVERTED INDEX & GRAPH ====================
with tab_index:
    st.markdown("### 📊 Vectorless Inverted Index & Knowledge Graph")
    st.caption("Explore how text is indexed without vector embeddings or dense dimensionality.")

    idx_col1, idx_col2 = st.columns([1, 1])
    with idx_col1:
        st.markdown("#### 📚 Inverted Index Vocabulary")
        top_terms = engine.indexer.get_vocabulary_stats(top_n=25)
        if top_terms:
            df_vocab = pd.DataFrame(top_terms)
            st.dataframe(df_vocab, use_container_width=True, hide_index=True)
        else:
            st.info("No documents currently indexed.")

        st.markdown("#### 🔎 Postings List Lookup")
        lookup_word = st.text_input("Look up word in inverted index", value="qubits")
        clean_word = lookup_word.lower().strip()
        if clean_word in engine.indexer.inverted_index:
            postings = engine.indexer.inverted_index[clean_word]
            st.success(f"Term `{clean_word}` found in {len(postings)} chunks:")
            post_data = []
            for cid, count in postings:
                chunk_obj = engine.indexer.chunk_map.get(cid)
                doc_name = chunk_obj.doc_name if chunk_obj else "Unknown"
                post_data.append({"Chunk ID": cid, "Document": doc_name, "Term Frequency": count})
            st.table(pd.DataFrame(post_data))
        else:
            st.caption(f"Word `{clean_word}` is not present in the current vocabulary.")

    with idx_col2:
        st.markdown("#### 🕸️ Extracted Entity Co-occurrence Network")
        top_ents = graph_stats.get("top_entities", [])
        if top_ents:
            df_ents = pd.DataFrame(top_ents)
            st.dataframe(df_ents, use_container_width=True, hide_index=True)
            st.caption(f"Network Summary: **{graph_stats['num_nodes']}** Entities, **{graph_stats['num_edges']}** Co-occurrence Connections.")
        else:
            st.info("No entities extracted yet.")


# ==================== TAB 4: DOCUMENT MANAGER ====================
with tab_docs:
    st.markdown("### 📁 Document Knowledge Base Manager")
    st.caption("Upload your own PDFs, Markdown, or Text files, or inspect active knowledge chunks.")

    upload_col1, upload_col2 = st.columns([1, 1])
    
    with upload_col1:
        st.markdown("#### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )
        if uploaded_files:
            for file in uploaded_files:
                file_bytes = file.read()
                if file.name.endswith(".pdf"):
                    n_chunks = engine.ingest_pdf(file_bytes, file.name)
                else:
                    text_content = file_bytes.decode("utf-8", errors="ignore")
                    n_chunks = engine.ingest_text(text_content, file.name)
                st.success(f"Ingested `{file.name}` into {n_chunks} granular chunks!")
            st.rerun()

    with upload_col2:
        st.markdown("#### 📝 Quick Text Paste")
        custom_doc_title = st.text_input("Document Name", value="Custom_Note.txt")
        custom_doc_text = st.text_area("Paste Content Here", height=140, placeholder="Paste articles, specifications, manuals...")
        if st.button("Ingest Pasted Content", use_container_width=True):
            if custom_doc_text.strip():
                n_c = engine.ingest_text(custom_doc_text, custom_doc_title)
                st.success(f"Ingested `{custom_doc_title}` with {n_c} chunks!")
                st.rerun()

    st.divider()

    st.markdown("#### 📑 Active Indexed Chunks")
    if engine.indexer.chunks:
        chunk_rows = []
        for c in engine.indexer.chunks:
            chunk_rows.append({
                "Chunk ID": c.chunk_id,
                "Document": c.doc_name,
                "Page": c.page_number,
                "Section": c.section_title,
                "Words": len(c.content.split()),
                "Preview": c.content[:100] + "..."
            })
        st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No documents are currently indexed.")
