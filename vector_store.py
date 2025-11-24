from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pickle
import math
import re

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


# Keywords used to adjust scoring when questions are about macro / policy topics
ECON_KEYWORDS = [
    "growth",
    "gdp",
    "inflation",
    "fiscal",
    "monetary",
    "policy",
    "projection",
    "forecast",
    "projected",
    "outlook",
    "economic outlook",
    "risks",
    "recommendations",
    "imf",
    "real gdp",
    "real gdp growth",
    "2024",
    "2025",
]

# Keywords that strongly suggest the user is asking for numeric / tabular data
NUMERIC_HINT_KEYWORDS = [
    "table",
    "tables",
    "statistics",
    "stats",
    "data",
    "dataset",
    "numeric",
    "numerical",
    "numbers",
    "figures",
    "values",
    "gdp table",
    "inflation table",
    "projection table",
    "forecast table",
    "projection",
    "forecast",
    "2024",
    "2025",
]


class VectorStore:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model_name: str = "sentence-transformers/ms-marco-MiniLM-L-6-v2",
    ):
        print(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = None
        self.chunks = []

        # BM25 state
        self.bm25_docs = []        # list[dict[token -> tf]]
        self.bm25_doc_lens = []    # list[int]
        self.bm25_avgdl = 0.0
        self.bm25_idf = {}         # dict[token -> idf]

        # Cross-encoder reranker
        self.cross_encoder = None
        self._init_cross_encoder(cross_encoder_model_name)

        print("successfully loaded")

    def _init_cross_encoder(self, model_name: str) -> None:
        if CrossEncoder is None:
            print(
                "Warning: sentence-transformers not installed; "
                "cross-encoder reranking will be disabled."
            )
            return
        try:
            self.cross_encoder = CrossEncoder(model_name)
            print(f"Loaded cross-encoder reranker: {model_name}")
        except Exception as e:
            print(f"Warning: could not load cross-encoder model {model_name}: {e}")
            self.cross_encoder = None

    # -----------------------------
    # BM25 keyword index
    # -----------------------------
    @staticmethod
    def _tokenize(text: str):
        text = text.lower()
        return [t for t in re.split(r"\W+", text) if t]

    def _build_bm25_index(self, chunks):
        """Build a simple BM25 index over all chunks' content."""
        docs_tf = []
        doc_lens = []
        df = {}

        for chunk in chunks:
            tokens = self._tokenize(chunk.get("content", ""))
            doc_len = len(tokens)
            doc_lens.append(doc_len)

            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            docs_tf.append(tf)

            for t in tf.keys():
                df[t] = df.get(t, 0) + 1

        self.bm25_docs = docs_tf
        self.bm25_doc_lens = doc_lens
        if len(doc_lens) > 0:
            self.bm25_avgdl = sum(doc_lens) / len(doc_lens)
        else:
            self.bm25_avgdl = 0.0

        N = len(docs_tf)
        idf = {}
        for t, df_t in df.items():
            # Standard BM25 idf
            idf[t] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        self.bm25_idf = idf

        print(f"BM25 index built for {N} chunks")

    def _bm25_search(self, query: str, top_n: int = 20, k1: float = 1.5, b: float = 0.75):
        """Return list of (chunk_id, bm25_score)."""
        if not self.bm25_docs or self.bm25_avgdl == 0.0:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        q_terms = set(q_tokens)
        scores = [0.0] * len(self.bm25_docs)

        for idx, tf in enumerate(self.bm25_docs):
            score = 0.0
            dl = self.bm25_doc_lens[idx]
            for t in q_terms:
                if t not in tf or t not in self.bm25_idf:
                    continue
                idf = self.bm25_idf[t]
                freq = tf[t]
                denom = freq + k1 * (1 - b + b * dl / self.bm25_avgdl)
                score += idf * freq * (k1 + 1) / denom
            scores[idx] = score

        # Get top_n docs by BM25 score
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(idx, s) for idx, s in ranked[:top_n] if s > 0]

    # -----------------------------
    # Query analysis and type bias
    # -----------------------------
    @staticmethod
    def _analyze_query(query: str):
        q_lower = query.lower()
        contains_econ = any(kw in q_lower for kw in ECON_KEYWORDS)

        is_numeric = any(kw in q_lower for kw in NUMERIC_HINT_KEYWORDS)
        # Also treat presence of explicit numbers as signal for numeric intent
        if not is_numeric:
            is_numeric = any(token.isdigit() for token in re.split(r"\W+", q_lower))

        forecast_terms = [
            "growth",
            "gdp",
            "forecast",
            "projection",
            "projected",
            "2024",
            "2025",
        ]
        has_forecast_keywords = any(kw in q_lower for kw in forecast_terms)

        return {
            "contains_econ_keywords": contains_econ,
            "is_numeric_question": is_numeric,
            "has_forecast_keywords": has_forecast_keywords,
        }

    @staticmethod
    def _compute_type_bias(chunk_type: str, query_info: dict) -> float:
        """
        Apply metadata-based weighting:
        - For macro/policy keywords: boost TEXT, demote TABLE/IMAGE.
        - For forecast questions with GDP/growth terms: strongly prefer narrative TEXT.
        - For conceptual questions: text > table > image.
        - For numeric questions (without forecast emphasis): prefer tables.
        """
        chunk_type = (chunk_type or "text").lower()
        bias = 0.0

        contains_econ = query_info.get("contains_econ_keywords", False)
        has_forecast = query_info.get("has_forecast_keywords", False)

        if contains_econ:
            if chunk_type == "text":
                bias += 0.3
            elif chunk_type in ("table", "image"):
                bias -= 0.3

        # For forecast-style economic questions, always prioritise narrative text
        if has_forecast:
            if chunk_type == "text":
                bias += 0.4
            elif chunk_type in ("table", "image"):
                bias -= 0.2
            # When forecast emphasis is present, skip further numeric/conceptual adjustment
            return bias

        # 3. Conceptual vs numeric weighting
        if query_info.get("is_numeric_question"):
            # Numeric question: prefer tables
            if chunk_type == "table":
                bias += 0.2
            elif chunk_type == "image":
                bias -= 0.1
        else:
            # Conceptual question: text > table > image
            if chunk_type == "text":
                bias += 0.2
            elif chunk_type == "image":
                bias -= 0.1

        return bias

    # -----------------------------
    # Keyword-based reranking helpers
    # -----------------------------
    @staticmethod
    def _keyword_boost_for_chunk(query: str, chunk: dict) -> float:
        """Lightweight keyword-based boost on top of vector and BM25 scores.

        - If the chunk contains numeric forecasts or projections -> +0.25
        - If it contains phrases like 'real GDP growth is projected' -> +0.4
        """
        text = (chunk.get("content") or "").lower()
        q_lower = query.lower()
        boost = 0.0

        # Numeric forecast / projection with numbers
        numeric_terms = ["forecast", "projection", "projected", "expected"]
        has_numeric_term = any(term in text for term in numeric_terms)
        has_number = any(ch.isdigit() for ch in text)
        if has_numeric_term and has_number:
            boost += 0.25

        # Explicit real GDP growth projections
        if "real gdp growth" in text or (
            "gdp" in text
            and "growth" in text
            and ("projected" in text or "forecast" in text or "projection" in text)
        ):
            boost += 0.4

        # Small extra tie-breaker when both query and chunk mention key GDP/forecast terms
        query_keywords = ["growth", "gdp", "forecast", "projection", "2024", "2025"]
        if any(kw in q_lower for kw in query_keywords) and any(
            kw in text for kw in query_keywords
        ):
            boost += 0.1

        return boost

    # -----------------------------
    # Embedding creation
    # -----------------------------
    def create_embeddings(self, chunks):
        """
        Build FAISS index and BM25 keyword index from prepared chunks.
        Each chunk is expected to have: content, page, type, section, source.
        """
        self.chunks = chunks
        documents = []

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    "page": chunk.get("page"),
                    "type": chunk.get("type"),
                    "section": chunk.get("section"),
                    "source": chunk.get("source"),
                    "chunk_id": i,
                },
            )
            documents.append(doc)

        print("Building FAISS index...")
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

        print(f"FAISS index with {len(documents)} vectors")

        # Build BM25 keyword index for hybrid search
        self._build_bm25_index(chunks)

    def _get_chunk_by_id(self, chunk_id: int):
        if 0 <= chunk_id < len(self.chunks):
            return self.chunks[chunk_id]
        return None

    # -----------------------------
    # Hybrid search + reranking
    # -----------------------------
    def search(self, query, k: int = 5):
        """
        Hybrid retrieval:
        1) FAISS vector search to get semantic candidates.
        2) BM25 keyword search for lexical candidates.
        3) Union candidates and rerank with a cross-encoder.
        4) Apply metadata-based type bias and macro-keyword adjustments.
        """
        if self.vectorstore is None:
            print("Vectorstore not created")
            return []

        # 1. Vector search (get more than k for better hybrid coverage)
        vector_k = max(k * 4, 20)
        faiss_results = self.vectorstore.similarity_search_with_score(
            query, k=vector_k
        )

        candidate_ids = set()
        vector_rank_map = {}

        for rank, (doc, _score) in enumerate(faiss_results):
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id is None:
                continue
            candidate_ids.add(chunk_id)
            # Store best rank for this chunk
            if chunk_id not in vector_rank_map:
                vector_rank_map[chunk_id] = rank

        # 2. BM25 keyword search
        bm25_k = max(k * 4, 20)
        bm25_results = self._bm25_search(query, top_n=bm25_k)
        bm25_rank_map = {}

        for rank, (chunk_id, _bm25_score) in enumerate(bm25_results):
            candidate_ids.add(chunk_id)
            if chunk_id not in bm25_rank_map:
                bm25_rank_map[chunk_id] = rank

        if not candidate_ids:
            return []

        # 3. Cross-encoder reranking (if available)
        candidates = []
        for chunk_id in candidate_ids:
            chunk = self._get_chunk_by_id(chunk_id)
            if not chunk:
                continue
            candidates.append((chunk_id, chunk["content"]))

        if not candidates:
            return []

        if self.cross_encoder is not None:
            pairs = [(query, text[:512]) for (_cid, text) in candidates]
            ce_scores = self.cross_encoder.predict(pairs)
        else:
            # No cross-encoder: use rank-based proxy scores from vector/BM25
            ce_scores = []
            for cid, _text in candidates:
                # Higher is better, so negate rank; smaller rank => larger value
                v_rank = vector_rank_map.get(cid, len(candidate_ids))
                b_rank = bm25_rank_map.get(cid, len(candidate_ids))
                proxy = -min(v_rank, b_rank)
                ce_scores.append(float(proxy))

        query_info = self._analyze_query(query)

        # Combine cross-encoder score + type bias + keyword-based boost
        scored = []
        for (cid, _text), ce_score in zip(candidates, ce_scores):
            chunk = self._get_chunk_by_id(cid)
            if not chunk:
                continue
            chunk_type = chunk.get("type", "text")
            type_bias = self._compute_type_bias(chunk_type, query_info)
            kw_boost = self._keyword_boost_for_chunk(query, chunk)
            final_score = float(ce_score) + type_bias + kw_boost
            scored.append((final_score, cid))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        formatted_results = []
        for rank, (score, cid) in enumerate(top, start=1):
            ch = self._get_chunk_by_id(cid)
            if not ch:
                continue
            formatted_results.append(
                {
                    "chunk": {
                        "content": ch["content"],
                        "page": ch.get("page"),
                        "type": ch.get("type"),
                        "section": ch.get("section"),
                        "source": ch.get("source"),
                    },
                    "score": float(score),
                    "rank": rank,
                }
            )

        return formatted_results

    # -----------------------------
    # Persistence
    # -----------------------------
    def save(self, filepath="vector_store"):
        if self.vectorstore is None:
            print("No vectorstore to save")
            return
        self.vectorstore.save_local(filepath)

        with open(f"{filepath}_chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, filepath="vector_store"):
        self.vectorstore = FAISS.load_local(
            filepath,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        with open(f"{filepath}_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        # Rebuild BM25 index after loading
        if self.chunks:
            self._build_bm25_index(self.chunks)

        print("Loaded vector store chunks")


if __name__ == "__main__":
    test_chunks = [
        {
            "content": "Qatar has strong economic growth",
            "page": 1,
            "type": "text",
            "section": "intro",
            "source": "Page 1",
        },
        {
            "content": "Banking sector remains healthy",
            "page": 2,
            "type": "text",
            "section": "banking",
            "source": "Page 2",
        },
        {
            "content": "IMF recommendations for fiscal policy",
            "page": 3,
            "type": "text",
            "section": "policy",
            "source": "Page 3",
        },
    ]

    print("Testing Vector Store with hybrid + reranking...")
    store = VectorStore()
    store.create_embeddings(test_chunks)

    results = store.search("What is Qatar's economic situation?", k=2)
    print("\nSearch Results:")
    for result in results:
        print(
            f"Rank {result['rank']}: {result['chunk']['content'][:50]}..."
            f" (Score: {result['score']:.3f})"
        )
