from vector_store import VectorStore
from llm_qa import LLMQA, SimpleQA
import config


def main():
    print("Loading vector store and QA model for a smoke test...")

    vs = VectorStore(model_name=config.EMBEDDING_MODEL)
    vs.load(config.VECTOR_STORE_PATH)

    try:
        qa = LLMQA(model_name=config.LLM_MODEL)
    except Exception as e:
        print("LLMQA failed, falling back to SimpleQA:", e)
        qa = SimpleQA()

    question = "What is Qatar's GDP growth forecast for 2024-25?"
    print("\nQUESTION:", question)

    results = vs.search(question, k=5)
    print("Retrieved", len(results), "chunks")
    for r in results[:3]:
        ch = r["chunk"]
        print(" - Rank", r["rank"], "score", r["score"], "type", ch.get("type"), "source", ch.get("source"))

    answer = qa.generate_answer_with_citations(question, results)
    print("\nANSWER:\n", answer["answer"])
    print("\nCITATIONS:")
    for c in answer["citations"]:
        print(c)


if __name__ == "__main__":
    main()
