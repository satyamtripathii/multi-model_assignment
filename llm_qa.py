from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re

FALLBACK_MESSAGE = (
    "This question may not be covered in the document. "
    "Try asking something related to economic outlook, fiscal policy, "
    "monetary policy, or NDS3 reforms."
)

# Keywords used when doing fuzzy relevance checks between question and chunks
ECON_RELATED_KEYWORDS = [
    "gdp",
    "growth",
    "forecast",
    "projection",
    "projected",
    "economic outlook",
    "outlook",
    "imf",
]

YEAR_KEYWORDS = ["2024", "2025"]


def _postprocess_answer_text(text: str) -> str:
    """Apply lightweight corrections to common numeric / fraction errors.

    In particular, guard against Unicode fraction expansion issues that can
    turn values like "5½ percent" into "512 percent".
    """
    if not text:
        return text

    # Fix specific hallucination/typo patterns requested
    text = re.sub(r"\b512 percent\b", "5.5 percent", text)
    text = re.sub(r"\b512%\b", "5.5%", text)

    return text


class LLMQA:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        print(f"Loading LLM model via LangChain: {model_name}")

        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                device=device,
                temperature=0.0,  # more deterministic for classification / grounding
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

            # New prompt: strictly use provided context; no chain-of-thought.
            self.prompt_template = (
                "You are a question-answering assistant for a single economic report.\n\n"
                "You must answer strictly and only from the CONTEXT below. "
                "If the context does not contain enough information to answer the question, "
                "reply exactly with:\n"
                "No evidence found in the document.\n\n"
                "Do NOT use any outside knowledge. "
                "Do NOT show your reasoning or chain-of-thought. "
                "Provide only a concise final answer.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer (one or two short paragraphs at most):"
            )

            print(f"LangChain LLM loaded on {device_name}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _build_context_text(
        self,
        context_chunks,
        max_chars_per_chunk: int = 500,
        max_chunks: int = 5,
    ) -> str:
        pieces = []
        for chunk in context_chunks[:max_chunks]:
            src = chunk.get("source", "Unknown source")
            page = chunk.get("page", "?")
            content = chunk.get("content", "")[:max_chars_per_chunk]
            pieces.append(f"[Source: {src} | Page: {page}]\n{content}")
        return "\n\n".join(pieces)

    def _keyword_match_score(self, query: str, chunk_text: str):
        """Return a fuzzy keyword-overlap score in [0, 1] plus keyword flags.

        This is purely lexical and is used to avoid false negatives when the
        embedding model or LLM underestimates relevance.
        """
        query_lower = (query or "").lower()
        text_lower = (chunk_text or "").lower()

        q_tokens = [t for t in re.split(r"\W+", query_lower) if t]
        t_tokens = [t for t in re.split(r"\W+", text_lower) if t]

        q_set = set(q_tokens)
        t_set = set(t_tokens)
        overlap = q_set.intersection(t_set)

        has_query_keywords = len(overlap) > 0

        # Base lexical overlap score (Jaccard-like)
        denom = max(len(q_set), 1)
        base_score = len(overlap) / denom

        # Strong economic terms: GDP, growth, IMF projections, outlook, years
        econ_hit_in_chunk = any(kw in text_lower for kw in ECON_RELATED_KEYWORDS)
        econ_hit_in_query = any(kw in query_lower for kw in ECON_RELATED_KEYWORDS)
        year_hit_in_query = any(y in query_lower for y in YEAR_KEYWORDS)
        year_hit_in_chunk = any(y in text_lower for y in YEAR_KEYWORDS)

        has_econ_or_year_hit = (econ_hit_in_chunk and econ_hit_in_query) or (
            year_hit_in_query and year_hit_in_chunk
        )

        # If both question and chunk mention key econ terms / years, treat this
        # as at least "PARTIAL" relevance even when lexical overlap is small.
        if has_econ_or_year_hit:
            base_score = max(base_score, 0.5)

        base_score = float(max(0.0, min(1.0, base_score)))
        return base_score, has_query_keywords, has_econ_or_year_hit

    def _llm_relevance_label(self, query: str, chunk: dict):
        """Ask the LLM for a soft relevance label: YES / PARTIAL / NO.

        Returns (label_str, score) where score is in {1.0, 0.5, 0.0}.
        """
        content = chunk.get("content", "")[:500]
        prompt = (
            "You are checking whether a document chunk answers a user's question.\n\n"
            f"Question:\n{query}\n\n"
            f"Chunk:\n{content}\n\n"
            "Does this chunk help answer the user's question? "
            "Reply with exactly one word: YES, PARTIAL, or NO."
        )
        try:
            response = self.llm.invoke(prompt).strip().upper()
        except Exception as e:
            print(f"Error during chunk relevance check: {e}")
            # If validation fails, fall back to neutral PARTIAL
            return "PARTIAL", 0.5

        if response.startswith("YES"):
            return "YES", 1.0
        if response.startswith("PART"):
            return "PARTIAL", 0.5
        if response.startswith("NO"):
            return "NO", 0.0

        # Unknown label -> treat as partial match
        return "PARTIAL", 0.5

    def _get_chunk_relevance_score(self, query: str, chunk: dict):
        """Combine LLM label and keyword overlap into a soft confidence score.

        The final score is max(llm_score, keyword_score) so that strong
        keyword matches are never discarded even if the LLM is conservative.
        """
        keyword_score, has_query_keywords, has_econ_or_year_hit = self._keyword_match_score(
            query, chunk.get("content", "")
        )
        label, llm_score = self._llm_relevance_label(query, chunk)

        final_score = max(llm_score, keyword_score)
        return final_score, {
            "label": label,
            "llm_score": llm_score,
            "keyword_score": keyword_score,
            "has_query_keywords": has_query_keywords,
            "has_econ_or_year_hit": has_econ_or_year_hit,
        }

    def _filter_relevant_results(self, query: str, search_results, max_to_check: int = 5):
        """Score top-k retrieved chunks and keep those above a soft threshold.

        Returns:
            filtered_results: list of results with score >= 0.5 (YES or PARTIAL)
            scores: list of float scores for the inspected results
            any_keyword_hit: True if any inspected chunk matched econ/query keywords
        """
        if not search_results:
            return [], [], False

        scored_results = []
        scores = []
        any_keyword_hit = False

        for i, result in enumerate(search_results[:max_to_check]):
            score, meta = self._get_chunk_relevance_score(query, result["chunk"])
            scored_results.append((result, score, meta))
            scores.append(score)
            if meta.get("has_query_keywords") or meta.get("has_econ_or_year_hit"):
                any_keyword_hit = True

        filtered_results = [res for (res, score, _meta) in scored_results if score >= 0.5]

        return filtered_results, scores, any_keyword_hit

    def _is_answer_grounded(self, context_text: str, answer: str) -> bool:
        """
        Validate that the final answer is grounded in the retrieved context.
        If not, we override with 'No evidence found in the document.'.
        """
        validation_prompt = (
            "You are a strict fact-checker for answers based on a document.\n\n"
            "Given the context and an answer, determine if the answer is fully supported "
            "by the context. If all important statements are directly supported or "
            "clearly paraphrased from the context, reply YES. Otherwise reply NO.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Answer:\n{answer}\n\n"
            "Reply with a single word: YES or NO."
        )

        try:
            resp = self.llm.invoke(validation_prompt).strip().upper()
        except Exception as e:
            print(f"Error during answer grounding check: {e}")
            # If validation fails, do not block the answer
            return True

        return resp.startswith("YES")

    # -----------------------------
    # Answer generation
    # -----------------------------
    def generate_answer(self, query, context_chunks):
        if not context_chunks:
            return "No evidence found in the document."

        context_text = self._build_context_text(context_chunks)

        prompt = self.prompt_template.format(
            context=context_text,
            question=query,
        )

        try:
            result = self.llm.invoke(prompt)
            answer = result.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Sorry, I encountered an error generating the answer."

        if not answer.strip():
            return "No evidence found in the document."

        # Apply numeric / fraction post-processing corrections
        answer = _postprocess_answer_text(answer)

        # Grounding is now enforced softly in generate_answer_with_citations,
        # where we may add a safety note instead of blocking the answer.
        return answer

    def generate_answer_with_citations(self, query, search_results):
        """Generate an answer plus citations using soft validation.

        Changes vs. the strict version:
        - Chunks are scored (0.0–1.0) instead of YES/NO.
        - Fallback is only triggered when the top 3 chunks all have confidence
          < 0.2 *and* no chunk matches econ/query keywords.
        - Otherwise we always answer using the best chunks and, if needed,
          append a safety note when confidence is low or grounding is weak.
        """
        if not search_results:
            return {
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "context_used": 0,
            }

        # Step 1: score and filter the top retrieved chunks
        filtered_results, relevance_scores, any_keyword_hit = self._filter_relevant_results(
            query, search_results, max_to_check=5
        )

        top3_scores = relevance_scores[:3]
        low_confidence_top3 = bool(top3_scores) and all(s < 0.2 for s in top3_scores)

        # Step 2: conservative fallback only when nothing looks relevant
        if low_confidence_top3 and not any_keyword_hit:
            return {
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "context_used": 0,
            }

        # If nothing passed the 0.5 threshold, still use top-k as weak context
        if not filtered_results:
            filtered_results = search_results[:3]

        context_chunks = [res["chunk"] for res in filtered_results]
        context_text = self._build_context_text(context_chunks)
        answer = self.generate_answer(query, context_chunks)

        # Soft confidence measure from validation scores
        avg_conf = (
            sum(relevance_scores) / len(relevance_scores)
            if relevance_scores
            else 1.0
        )

        safety_note = ""
        if avg_conf < 0.4:
            safety_note = (
                "\n\n_NOTE: The document evidence for this question is limited or "
                "partially relevant, so the answer may be incomplete._"
            )

        # Optional grounding check. If grounding is weak, we do NOT block the
        # answer; we just append a safety disclaimer.
        if not self._is_answer_grounded(context_text, answer):
            if not safety_note:
                safety_note = (
                    "\n\n_NOTE: This answer may not be fully supported by the "
                    "retrieved context and should be interpreted with caution._"
                )

        # Apply numeric / fraction post-processing corrections again at the
        # very end so any templated note also gets cleaned if needed.
        final_answer = _postprocess_answer_text(answer)
        if safety_note:
            final_answer = final_answer + safety_note

        citations = []
        for i, result in enumerate(filtered_results[:3]):
            chunk = result["chunk"]
            citations.append(
                {
                    "rank": i + 1,
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "type": chunk.get("type"),
                    "relevance_score": result.get("score", 0.0),
                }
            )

        return {
            "answer": final_answer,
            "citations": citations,
            "context_used": len(context_chunks),
            "confidence": float(avg_conf),
        }


class SimpleQA:
    def __init__(self):
        print()

    def generate_answer_with_citations(self, query, search_results):
        if not search_results:
            return {
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "context_used": 0,
            }

        top_chunks = search_results[:3]

        answer_parts = []
        for result in top_chunks:
            chunk = result["chunk"]
            snippet = chunk["content"][:200].strip()
            if snippet:
                answer_parts.append(f"From {chunk['source']}: {snippet}...")

        answer = (
            "\n\n".join(answer_parts)
            if answer_parts
            else "No relevant information found in the document."
        )

        citations = []
        for i, result in enumerate(top_chunks):
            chunk = result["chunk"]
            citations.append(
                {
                    "rank": i + 1,
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "type": chunk.get("type"),
                    "relevance_score": result.get("score", 0.0),
                }
            )

        return {
            "answer": answer,
            "citations": citations,
            "context_used": len(search_results),
        }


if __name__ == "__main__":

    test_results = [
        {
            "chunk": {
                "content": "Qatar economy grew by 5% in 2024 driven by strong non-hydrocarbon sector growth.",
                "page": 1,
                "type": "text",
                "section": "summary",
                "source": "Page 1",
            },
            "score": 0.85,
        },
        {
            "chunk": {
                "content": "The banking sector remains healthy with strong capital ratios.",
                "page": 2,
                "type": "text",
                "section": "banking",
                "source": "Page 2",
            },
            "score": 0.72,
        },
    ]
    try:
        print("\n1. Test LLMQA with self-validation and grounding")
        qa = LLMQA()
        result = qa.generate_answer_with_citations(
            "What is Qatar's growth?", test_results
        )
        print(f"\nAnswer: {result['answer']}")
        print(f"Citations: {len(result['citations'])} sources")

    except Exception as e:
        print(f"\nLangChain LLMQA failed: {e}")
        print("\n2. Test Fallback SimpleQA")
        qa = SimpleQA()
        result = qa.generate_answer_with_citations(
            "What is Qatar's growth?", test_results
        )
        print(f"\nAnswer: {result['answer']}")
        print(f"Citations: {len(result['citations'])} sources")
