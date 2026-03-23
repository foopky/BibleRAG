import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()


def main() -> None:
    index_name = os.getenv("PINECONE_INDEX", "bible-rag")

    print(
        {
            "has_openai": bool(os.getenv("OPENAI_API_KEY")),
            "has_pinecone": bool(os.getenv("PINECONE_API_KEY")),
            "index": index_name,
        }
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    print(index.describe_index_stats())

    store = PineconeVectorStore(
        index_name=index_name,
        embedding=OpenAIEmbeddings(),
        namespace="bible",
    )

    queries = [
        "Where is the verse about God creating the heavens and the earth in the beginning?",
        "What does Genesis 1 say about creation?",
    ]

    for query in queries:
        print(f"QUERY: {query}")
        results = store.similarity_search_with_score(query, k=3)
        print(f"RESULT_COUNT: {len(results)}")
        for index_number, (doc, score) in enumerate(results, start=1):
            print(
                "DOC",
                index_number,
                {
                    "score": score,
                    "reference": doc.metadata.get("reference"),
                    "chapter_id": doc.metadata.get("chapter_id"),
                    "chars": len(doc.page_content),
                },
            )
            print(doc.page_content[:300].replace("\n", " "))
        print("-" * 40)


if __name__ == "__main__":
    main()