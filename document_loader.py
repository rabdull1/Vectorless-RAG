"""
Document Loader module for Vectorless RAG.
Supports PDF, TXT, Markdown, CSV, and raw text strings.
Preserves structural metadata such as page numbers, section headers, and chunk indices.
"""

import io
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    import pypdf
except ImportError:
    pypdf = None


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_name: str
    content: str
    page_number: int
    section_title: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def source_tag(self) -> str:
        sec = f" | Section: {self.section_title}" if self.section_title != "General" else ""
        return f"[{self.doc_name} | Page {self.page_number}{sec} | Chunk {self.chunk_id}]"


class DocumentLoader:
    def __init__(self, chunk_size: int = 180, chunk_overlap: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Splits text respecting section headers and paragraphs.
        """
        # First split by major markdown headers if present
        header_sections = re.split(r'(?=\n#{1,3}\s+)', text)
        chunks = []

        for sec in header_sections:
            sec = sec.strip()
            if not sec:
                continue

            paragraphs = re.split(r'\n\s*\n', sec)
            current_chunk = []
            current_len = 0

            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue
                words = p.split()
                p_len = len(words)

                if current_len + p_len <= self.chunk_size:
                    current_chunk.append(p)
                    current_len += p_len
                else:
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    
                    if p_len > self.chunk_size:
                        sentences = re.split(r'(?<=[.?!])\s+', p)
                        sub_chunk = []
                        sub_len = 0
                        for s in sentences:
                            s_len = len(s.split())
                            if sub_len + s_len > self.chunk_size and sub_chunk:
                                chunks.append(" ".join(sub_chunk))
                                sub_chunk = [s]
                                sub_len = s_len
                            else:
                                sub_chunk.append(s)
                                sub_len += s_len
                        if sub_chunk:
                            current_chunk = [" ".join(sub_chunk)]
                            current_len = sum(len(s.split()) for s in current_chunk)
                        else:
                            current_chunk = []
                            current_len = 0
                    else:
                        current_chunk = [p]
                        current_len = p_len

            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [text]

    def _extract_section_header(self, text: str) -> str:
        """Heuristic to identify headings or prominent titles at chunk start."""
        first_line = text.strip().split("\n")[0]
        # Check markdown header
        md_match = re.match(r'^#{1,4}\s+(.+)$', first_line)
        if md_match:
            return md_match.group(1).strip()
        # Check uppercase title or short header line (< 60 chars)
        if len(first_line) < 60 and (first_line.isupper() or first_line.endswith(":")):
            return first_line.rstrip(":").strip()
        return "General"

    def load_pdf(self, file_bytes: bytes, filename: str) -> List[DocumentChunk]:
        """Extract pages from PDF and segment into chunks with page numbers."""
        if pypdf is None:
            raise ImportError("pypdf is required to process PDF files. Please install pypdf.")

        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if not page_text:
                continue

            page_chunks = self.split_text(page_text)
            for sub_text in page_chunks:
                chunk_idx += 1
                sec = self._extract_section_header(sub_text)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"c{chunk_idx}",
                        doc_name=filename,
                        content=sub_text,
                        page_number=page_num,
                        section_title=sec,
                        metadata={"type": "pdf", "total_pages": len(reader.pages)}
                    )
                )

        return chunks

    def load_text(self, text: str, filename: str) -> List[DocumentChunk]:
        """Loads plain text or markdown content into structured chunks."""
        chunks: List[DocumentChunk] = []
        raw_chunks = self.split_text(text)

        current_section = "Introduction"
        for i, chunk_text in enumerate(raw_chunks, start=1):
            sec_header = self._extract_section_header(chunk_text)
            if sec_header != "General":
                current_section = sec_header

            # Approximate page (1 page ~ 400 words)
            approx_page = max(1, (i * self.chunk_size) // 400 + 1)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"c{i}",
                    doc_name=filename,
                    content=chunk_text,
                    page_number=approx_page,
                    section_title=current_section,
                    metadata={"type": "text"}
                )
            )
        return chunks
