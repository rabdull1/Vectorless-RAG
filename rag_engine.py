"""
Vectorless RAG Engine.
Coordinates:
1. Document ingestion and vectorless indexing
2. Multi-strategy retrieval (BM25, Graph-Boosted, Hierarchical)
3. Grounded answer synthesis via Google Gemini API (gemini-3.8-flash)
4. Streaming response generator and citation verification
"""

import os
import time
from typing import List, Dict, Any, Optional, Generator, Tuple
from dotenv import load_dotenv

from document_loader import DocumentLoader, DocumentChunk
from indexer import VectorlessIndex

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


SYSTEM_PROMPT = """You are an advanced, hyper-accurate Vectorless RAG Assistant powered by Google Gemini.
Your answers MUST be strictly grounded in the provided document context chunks.

CRITICAL INSTRUCTIONS:
1. Grounding & Zero Hallucination: Use ONLY the information provided in the context chunks below. Do not guess, extrapolate, or assume outside facts. If the information is not contained in the context, explicitly state: "The provided documents do not contain sufficient information to answer this question."
2. Exact Precision: Pay strict attention to exact numbers, percentages, dates, acronyms, and technical terminology. Vectorless RAG is built for exact lexical precision.
3. Explicit Citations: For every claim, fact, or statistic you state, provide the exact source tag at the end of the sentence or bullet, in the format: `[DocName | Page X | Chunk cY]`.
4. Tone & Formatting: Provide clear, well-structured markdown answers with bullet points and bold highlights for readability.
"""


AVAILABLE_MODELS = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-pro-latest"
]

class VectorlessRAGEngine:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-flash-latest"):
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.loader = DocumentLoader(chunk_size=180, chunk_overlap=30)
        self.indexer = VectorlessIndex(k1=1.5, b=0.75)
        self.client: Optional[Any] = None
        self._init_gemini_client()

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self._init_gemini_client()

    def set_model(self, model_name: str):
        self.model_name = model_name

    def _init_gemini_client(self):
        if self.api_key and genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None
        else:
            self.client = None

    def ingest_text(self, text: str, filename: str) -> int:
        """Ingests plain text or markdown and rebuilds vectorless indexes."""
        new_chunks = self.loader.load_text(text, filename)
        all_chunks = [c for c in self.indexer.chunks if c.doc_name != filename] + new_chunks
        self.indexer.build(all_chunks)
        return len(new_chunks)

    def ingest_pdf(self, file_bytes: bytes, filename: str) -> int:
        """Ingests PDF bytes and rebuilds vectorless indexes."""
        new_chunks = self.loader.load_pdf(file_bytes, filename)
        all_chunks = [c for c in self.indexer.chunks if c.doc_name != filename] + new_chunks
        self.indexer.build(all_chunks)
        return len(new_chunks)

    def clear_documents(self):
        """Clears all indexed documents."""
        self.indexer.build([])

    def retrieve(self, query: str, mode: str = "BM25", top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves top relevant chunks vectorlessly.
        Modes: 'BM25', 'Graph-Boosted BM25', 'Hierarchical Section'
        """
        if not self.indexer.chunks:
            return []

        if mode == "Graph-Boosted BM25":
            return self.indexer.search_graph_expanded(query, top_k=top_k)
        elif mode == "Hierarchical Section":
            base_results = self.indexer.search_bm25(query, top_k=top_k)
            for res in base_results:
                chunk: DocumentChunk = res["chunk"]
                sec_chunks = self.indexer.hierarchy[chunk.doc_name].get(chunk.section_title, [])
                res["section_context"] = f"Section '{chunk.section_title}' ({len(sec_chunks)} chunks)"
            return base_results
        else:
            return self.indexer.search_bm25(query, top_k=top_k)

    def _build_prompt(self, query: str, retrieved_items: List[Dict[str, Any]]) -> Tuple[str, str]:
        context_parts = []
        for i, item in enumerate(retrieved_items, start=1):
            chunk: DocumentChunk = item["chunk"]
            score_info = f"BM25 Score: {item['score']}"
            if "graph_boost" in item and item["graph_boost"] > 0:
                score_info += f" (Base: {item['base_bm25']} + Graph: {item['graph_boost']})"
            
            header = f"--- CONTEXT CHUNK #{i} {chunk.source_tag} [{score_info}] ---"
            context_parts.append(f"{header}\n{chunk.content}\n")

        full_context = "\n".join(context_parts)
        user_prompt = f"""DOCUMENT CONTEXT:
{full_context}

USER QUESTION:
{query}

Please answer the user question thoroughly based strictly on the above document context. Include citations `[DocName | Page X | Chunk cY]` for all assertions."""
        return full_context, user_prompt

    def generate_answer_stream(
        self,
        query: str,
        retrieval_mode: str = "BM25",
        top_k: int = 4,
        temperature: float = 0.2
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streams answer tokens from Gemini while yielding retrieval metadata upfront.
        Yields:
          - First: {"type": "metadata", "retrieved_chunks": [...], "retrieval_time": float, "raw_context": str}
          - Next chunks: {"type": "token", "text": str}
          - End: {"type": "done", "total_time": float}
        """
        start_time = time.time()
        retrieved_items = self.retrieve(query, mode=retrieval_mode, top_k=top_k)
        retrieval_time = round(time.time() - start_time, 3)

        if not retrieved_items:
            yield {
                "type": "metadata",
                "retrieved_chunks": [],
                "retrieval_time": retrieval_time,
                "raw_context": ""
            }
            yield {
                "type": "token",
                "text": "No relevant documents found. Please ingest documents or verify your search terms."
            }
            yield {"type": "done", "total_time": retrieval_time}
            return

        full_context, user_prompt = self._build_prompt(query, retrieved_items)

        yield {
            "type": "metadata",
            "retrieved_chunks": retrieved_items,
            "retrieval_time": retrieval_time,
            "raw_context": full_context
        }

        if not self.client:
            yield {
                "type": "token",
                "text": "Gemini API Client is not configured. Please enter a valid GEMINI_API_KEY in the sidebar."
            }
            yield {"type": "done", "total_time": round(time.time() - start_time, 3)}
            return

        # Attempt call with up to 3 retries on transient 503 errors
        max_retries = 3
        stream = None
        for attempt in range(max_retries):
            try:
                stream = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=temperature,
                        max_output_tokens=2048,
                    )
                )
                break
            except Exception as e:
                err_str = str(e)
                if "503" in err_str and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    yield {"type": "token", "text": f"\n\n*Error contacting Gemini API: {err_str}*"}
                    yield {"type": "done", "total_time": round(time.time() - start_time, 3)}
                    return

        if stream:
            try:
                for chunk in stream:
                    if chunk.text:
                        yield {"type": "token", "text": chunk.text}
            except Exception as e:
                yield {"type": "token", "text": f"\n\n*Stream interrupted: {str(e)}*"}

        yield {"type": "done", "total_time": round(time.time() - start_time, 3)}

    def generate_answer(
        self, 
        query: str, 
        retrieval_mode: str = "BM25", 
        top_k: int = 4,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """Synchronous version returning full answer."""
        start_time = time.time()
        retrieved_items = self.retrieve(query, mode=retrieval_mode, top_k=top_k)
        retrieval_time = round(time.time() - start_time, 3)

        if not retrieved_items:
            return {
                "answer": "No relevant documents found. Please ingest documents or verify your query terms.",
                "retrieved_chunks": [],
                "retrieval_time": retrieval_time,
                "generation_time": 0.0,
                "model_used": self.model_name,
                "raw_context": ""
            }

        full_context, user_prompt = self._build_prompt(query, retrieved_items)

        if not self.client:
            return {
                "answer": "Gemini API Client is not configured. Please supply a valid GEMINI_API_KEY in the sidebar or `.env` file.",
                "retrieved_chunks": retrieved_items,
                "retrieval_time": retrieval_time,
                "generation_time": 0.0,
                "model_used": self.model_name,
                "raw_context": full_context
            }

        gen_start = time.time()
        answer_text = ""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=temperature,
                        max_output_tokens=2048,
                    )
                )
                answer_text = response.text or "No response generated by model."
                break
            except Exception as e:
                if "503" in str(e) and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                answer_text = f"Gemini API Error: {str(e)}"
                break

        gen_time = round(time.time() - gen_start, 3)

        return {
            "answer": answer_text,
            "retrieved_chunks": retrieved_items,
            "retrieval_time": retrieval_time,
            "generation_time": gen_time,
            "model_used": self.model_name,
            "raw_context": full_context
        }
