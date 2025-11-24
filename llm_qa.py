from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

FALLBACK_MESSAGE = (
    "This question may not be covered in the document. "
    "Try asking something related to economic outlook, fiscal policy, "
    "monetary policy, or NDS3 reforms."
)


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

    def _is_chunk_relevant(self, query: str, chunk: dict) -> bool:
        """
        LLM-based self-validation of individual chunks.
        Asks: 'Does this chunk directly answer the users question? Reply YES or NO.'
        """
        content = chunk.get("content", "")[:500]
        prompt = (
            "You are checking whether a document chunk directly answers a user's question.\n\n"
            f"Question:\n{query}\n\n"
            f"Chunk:\n{content}\n\n"
            "Does this chunk directly answer the user's question? "
            "Reply with a single word: YES or NO."
        )
        try:
            response = self.llm.invoke(prompt).strip().upper()
        except Exception as e:
            print(f"Error during chunk relevance check: {e}")
            # If validation fails, err on the side of keeping the chunk
            return True

        return response.startswith("YES")

    def _filter_relevant_results(self, query: str, search_results, max_to_check: int = 5):
        """
        Run LLM-based self-validation over top-k retrieved chunks.
        Returns (filtered_results, relevance_flags_for_checked).
        """
        if not search_results:
            return [], []

        filtered = []
        relevance_flags = []

        for i, result in enumerate(search_results[:max_to_check]):
            is_rel = self._is_chunk_relevant(query, result["chunk"])
            relevance_flags.append(is_rel)
            if is_rel:
                filtered.append(result)

        return filtered, relevance_flags

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

        # LLM-based post-hoc validation for grounding
        if not self._is_answer_grounded(context_text, answer):
            return "No evidence found in the document."

        return answer

    def generate_answer_with_citations(self, query, search_results):
        """
        1) Self-validate top 5 chunks using LLM (YES/NO).
        2) If top 3 are all irrelevant -> fallback message.
        3) Use only validated chunks as context for answering.
        4) Validate final answer is grounded in context.
        """
        if not search_results:
            return {
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "context_used": 0,
            }

        # Step 1: self-validate top 5 chunks
        filtered_results, relevance_flags = self._filter_relevant_results(
            query, search_results, max_to_check=5
        )

        # Step 2: fallback rule if top 3 chunks are irrelevant
        top3_flags = relevance_flags[:3]
        if top3_flags and not any(top3_flags):
            return {
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "context_used": 0,
            }

        # If nothing passed validation, still use top-k as weak context
        if not filtered_results:
            filtered_results = search_results[:5]

        context_chunks = [res["chunk"] for res in filtered_results]
        answer = self.generate_answer(query, context_chunks)

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
            "answer": answer,
            "citations": citations,
            "context_used": len(context_chunks),
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
