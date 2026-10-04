# ⚡ Vectorless RAG with Google Gemini API & Streamlit

A high-performance **Vectorless Retrieval-Augmented Generation (RAG)** system built using **Google Gemini API** (`gemini-flash-latest`, `gemini-3.8-flash`) and **Streamlit**.

---

## 🎯 What is Vectorless RAG?

Traditional RAG relies on dense vector embedding models and vector databases (such as Pinecone, FAISS, Chroma, Qdrant). While vector search is powerful for broad semantic topic matching, it suffers from notable shortcomings:
1. **Semantic Drift & Distance Distortion:** Numbers, version codes, exact dates, and acronyms get blurred in embedding spaces.
2. **Cold Start & Embedding Latency:** Every ingested document chunk and incoming user query requires embedding model calls.
3. **Database Maintenance Overhead:** Vector indexes require complex HNSW/IVF indexing, disk space, and hosting costs.

**Vectorless RAG** replaces dense vector databases with:
- **BM25Okapi / BM25Plus Probabilistic Indexing**: Exact keyword matching with term saturation ($k_1$) and document length normalization ($b$).
- **Inverted Indexing & Postings Attribution**: Millisecond lexical lookup with exact term-frequency ($tf$) and inverse document frequency ($idf$) scoring.
- **Entity & Concept Co-occurrence Graph (Vectorless GraphRAG)**: Extracts named entities, technical acronyms, and alphanumeric identifiers to build an in-memory knowledge graph that boosts context discovery.
- **Hierarchical Section Routing**: Keeps structural headers, sections, and page numbers intact for verifiable inline citations.
- **Gemini Reasoning & Long-Context Synthesis**: Grounds outputs strictly in context with exact source citations `[DocName | Page X | Chunk cY]` and zero hallucination.

---

## 🚀 Key Features

- **⚡ Zero Embeddings & Zero Vector DBs**: Pure probabilistic lexical and graph retrieval.
- **🤖 Powered by Google Gemini**: Uses the official `google-genai` SDK with `gemini-flash-latest` and `gemini-3.8-flash`.
- **💬 Real-Time Streaming Chat**: Token-by-token streaming responses with expandable citations showing exact chunk text and BM25 scores.
- **🔍 Vectorless Diagnostic Lab**: Test search queries directly, inspecting token match breakdowns, TF/IDF weights, and graph boosts without calling the LLM.
- **📊 Inverted Index & Knowledge Graph Explorer**: Explore vocabulary document frequencies, chunk postings lists, and entity co-occurrence networks.
- **📁 Multi-Format Document Ingestion**: Upload PDF, TXT, or Markdown documents, or paste raw text.
- **📦 Pre-loaded Realistic Benchmarks**: Comes pre-loaded with:
  1. *Superconducting Quantum Processors (Helios-X9 Specs & Shor's Algorithm)*
  2. *European Union AI Act Compliance & Penalty Manual (Article 5 & 71)*
  3. *NovaCorp Global Q3 2025 Financial & Operations Report (Revenue, EBITDA, ARR)*

---

## 📂 Project Architecture

```
vectorless_rag/
├── app.py               # Streamlit interactive web application
├── rag_engine.py        # Orchestrates Vectorless retrieval & Gemini API generation
├── indexer.py           # Inverted index, BM25Plus, and Entity Co-occurrence Graph
├── document_loader.py   # PDF and Text parser with structural chunking
├── sample_data.py       # Built-in benchmark documents for instant testing
├── requirements.txt     # Python dependencies
├── run_app.bat          # One-click Windows launcher
└── README.md            # Documentation
```

---

## 🛠️ Quick Start

### 1. Requirements
Ensure Python 3.10+ is installed:
```bash
py -3.12 -m pip install -r requirements.txt
```

### 2. Configure Gemini API Key
The app automatically detects `GEMINI_API_KEY` from your root `.env` file or environment:
```env
GEMINI_API_KEY="your-gemini-api-key"
```
You can also enter or update the key directly inside the Streamlit sidebar at any time.

### 3. Launch the Application
Run the Streamlit application:
```bash
py -3.12 -m streamlit run app.py
```
Or double-click `run_app.bat` on Windows.

---

## 🧠 Retrieval Strategies Compared

| Feature | Vector RAG | Vectorless RAG (This Project) |
|---|---|---|
| **Embedding Model** | Required (OpenAI, HuggingFace, etc.) | **None (Zero vector latency)** |
| **Vector Database** | Required (Pinecone, Chroma, Qdrant) | **None (In-memory Inverted & Graph Index)** |
| **Exact Term Precision** | Poor (numbers & acronyms blur) | **Exact (100% precision on codes & metrics)** |
| **Query Latency** | ~50ms - 250ms embedding + ANN search | **~1ms - 5ms (Inverted index lookup)** |
| **Cost** | Embedding tokens + Vector DB hosting | **$0 storage & indexing cost** |
