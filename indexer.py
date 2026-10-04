"""
Vectorless Indexer module.
Implements:
1. Inverted Index (Vocab, Term Frequencies, Postings Lists)
2. BM25Okapi Probabilistic Search with Term Attribution
3. Entity & Concept Co-occurrence Graph (Vectorless GraphRAG)
4. Hierarchical Document / Section Navigation Index
"""

import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Any, Optional

import networkx as nx
from rank_bm25 import BM25Plus

from document_loader import DocumentChunk

# Common English stopwords
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", 
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", 
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", 
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", 
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", 
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", 
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", 
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", 
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", 
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", 
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", 
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", 
    "your", "yours", "yourself", "yourselves", "also", "thus", "therefore"
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenizes text preserving technical tokens, numbers, and acronyms.
    E.g., 'EBITDA', 'Q3', 'ISO-27001', 'O(log n)', '99.9%'.
    """
    # Find alphanumeric words, hyphenated compounds, and percentages
    tokens = re.findall(r'\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*%?\b', text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def extract_entities(text: str) -> List[str]:
    """
    Extracts high-value entities and technical concepts without vector models.
    Matches capitalized multi-word phrases, acronyms, and alphanumeric identifiers.
    """
    patterns = [
        # Capitalized multi-word proper entities: 'Quantum Key Distribution', 'European Union'
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
        # Acronyms & codes: 'EBITDA', 'API', 'LLM', 'Q3', 'AES-256', 'BM25'
        r'\b[A-Z0-9]{2,}(?:-[A-Z0-9]+)*\b',
        # Quoted terms: "Vectorless RAG"
        r'\"([A-Za-z0-9\s_-]+)\"'
    ]
    entities = set()
    for pat in patterns:
        for match in re.finditer(pat, text):
            cand = match.group(0).strip(' "')
            if len(cand) >= 2 and cand.lower() not in STOPWORDS:
                entities.add(cand)
    return list(entities)


class VectorlessIndex:
    """
    Complete Vectorless Inverted and Graph-based Index.
    Does NOT require any vector embedding models or vector databases.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.chunk_map: Dict[str, DocumentChunk] = {}
        
        # Inverted index: term -> list of (chunk_id, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # Document frequencies: term -> number of chunks containing term
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        # Tokenized corpora for BM25
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Plus] = None
        
        # Knowledge Graph (Entities & Concept Co-occurrences)
        self.entity_graph: nx.Graph = nx.Graph()
        # Entity to chunk IDs
        self.entity_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        
        # Hierarchical Table of Contents: doc_name -> list of sections
        self.hierarchy: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    def build(self, chunks: List[DocumentChunk]):
        """Builds all vectorless indexes from chunks."""
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.inverted_index.clear()
        self.doc_freqs.clear()
        self.tokenized_corpus.clear()
        self.entity_graph.clear()
        self.entity_to_chunks.clear()
        self.hierarchy.clear()

        for chunk in chunks:
            # 1. Tokenize chunk content
            tokens = tokenize(chunk.content)
            self.tokenized_corpus.append(tokens)
            
            # 2. Inverted index postings
            tf_counter = Counter(tokens)
            for term, count in tf_counter.items():
                self.inverted_index[term].append((chunk.chunk_id, count))
                self.doc_freqs[term] += 1
                
            # 3. Hierarchy
            self.hierarchy[chunk.doc_name][chunk.section_title].append(chunk.chunk_id)
            
            # 4. Extract entities & populate vectorless graph
            entities = extract_entities(chunk.content)
            for ent in entities:
                self.entity_to_chunks[ent].add(chunk.chunk_id)
                if not self.entity_graph.has_node(ent):
                    self.entity_graph.add_node(ent, count=1, docs={chunk.doc_name})
                else:
                    self.entity_graph.nodes[ent]['count'] += 1
                    self.entity_graph.nodes[ent]['docs'].add(chunk.doc_name)
                    
            # Add co-occurrence edges
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1, e2 = entities[i], entities[j]
                    if self.entity_graph.has_edge(e1, e2):
                        self.entity_graph[e1][e2]['weight'] += 1
                    else:
                        self.entity_graph.add_edge(e1, e2, weight=1)

        # 5. Initialize BM25Plus
        if self.tokenized_corpus:
            self.bm25 = BM25Plus(self.tokenized_corpus, k1=self.k1, b=self.b)

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Calculates BM25 scores and provides term-level attribution.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        
        # Rank by score descending
        ranked_indices = sorted(
            range(len(raw_scores)), 
            key=lambda idx: raw_scores[idx], 
            reverse=True
        )

        results = []
        num_docs = len(self.chunks)

        for idx in ranked_indices[:top_k]:
            score = float(raw_scores[idx])
            if score <= 0:
                continue

            chunk = self.chunks[idx]
            chunk_tokens = self.tokenized_corpus[idx]
            chunk_tf = Counter(chunk_tokens)

            # Calculate individual term contributions
            term_matches = {}
            for q_term in set(query_tokens):
                tf = chunk_tf.get(q_term, 0)
                if tf > 0:
                    df = self.doc_freqs.get(q_term, 0)
                    # Standard BM25 IDF
                    idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
                    term_matches[q_term] = {
                        "tf": tf,
                        "df": df,
                        "idf": round(idf, 3)
                    }

            results.append({
                "chunk": chunk,
                "score": round(score, 4),
                "matched_terms": term_matches,
                "rank": len(results) + 1
            })

        return results

    def search_graph_expanded(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Expands query by traversing the entity co-occurrence graph.
        Finds connected concepts and boosts chunks containing related entities.
        """
        bm25_results = self.search_bm25(query, top_k=top_k * 2)
        if not bm25_results:
            return []

        query_entities = extract_entities(query)
        query_tokens = set(tokenize(query))

        # Find directly linked graph neighbors
        expanded_concepts = set()
        for ent in query_entities:
            if self.entity_graph.has_node(ent):
                neighbors = sorted(
                    self.entity_graph[ent].items(), 
                    key=lambda item: item[1]['weight'], 
                    reverse=True
                )
                for neighbor, edge_data in neighbors[:3]:
                    expanded_concepts.add((neighbor, edge_data['weight']))

        # Adjust score with graph co-occurrence bonus
        scored_candidates = []
        for res in bm25_results:
            chunk = res["chunk"]
            base_score = res["score"]
            graph_boost = 0.0
            found_graph_links = []

            for concept, weight in expanded_concepts:
                if concept.lower() in chunk.content.lower():
                    bonus = 0.5 * math.log1p(weight)
                    graph_boost += bonus
                    found_graph_links.append(f"{concept} (+{round(bonus, 2)})")

            final_score = base_score + graph_boost
            scored_candidates.append({
                "chunk": chunk,
                "score": round(final_score, 4),
                "base_bm25": base_score,
                "graph_boost": round(graph_boost, 4),
                "graph_links": found_graph_links,
                "matched_terms": res["matched_terms"]
            })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        for i, item in enumerate(scored_candidates[:top_k], start=1):
            item["rank"] = i

        return scored_candidates[:top_k]

    def get_vocabulary_stats(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Returns highest frequency terms across the vectorless inverted index."""
        sorted_terms = sorted(self.doc_freqs.items(), key=lambda x: x[1], reverse=True)[:top_n]
        stats = []
        total_chunks = max(1, len(self.chunks))
        for term, df in sorted_terms:
            idf = math.log((total_chunks - df + 0.5) / (df + 0.5) + 1.0)
            stats.append({
                "term": term,
                "doc_freq": df,
                "idf": round(idf, 3),
                "postings_count": len(self.inverted_index[term])
            })
        return stats

    def get_graph_summary(self) -> Dict[str, Any]:
        """Returns statistics for the vectorless entity graph."""
        top_nodes = sorted(
            self.entity_graph.nodes(data=True), 
            key=lambda x: x[1].get('count', 0), 
            reverse=True
        )[:15]
        
        return {
            "num_nodes": self.entity_graph.number_of_nodes(),
            "num_edges": self.entity_graph.number_of_edges(),
            "top_entities": [
                {"name": n[0], "mentions": n[1].get('count', 0), "docs": list(n[1].get('docs', []))}
                for n in top_nodes
            ]
        }
