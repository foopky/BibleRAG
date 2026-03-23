import os
import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "bible-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "bible")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
MIN_CONTEXT_CHARS = int(os.getenv("MIN_CONTEXT_CHARS", "120"))
RAG_DEBUG = os.getenv("RAG_DEBUG", "false").lower() in ("1", "true", "yes")

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful Bible scholar assistant. "
        "Answer using the provided Bible passages as your primary evidence. "
        "If retrieved passages are somewhat related, provide the best supported answer and explain limits briefly. "
        "Only say the context is insufficient when the context is actually empty. "
        "Always cite Bible references (for example: Genesis 1).\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite user questions into a concise English Bible retrieval query. "
        "Keep key entities, verse hints, and theological intent. "
        "Return only the rewritten query without explanations.",
    ),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('reference', doc.metadata.get('chapter_id', ''))}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(llm):
    return RAG_PROMPT | llm | StrOutputParser()


def build_rewriter_chain(llm):
    return REWRITE_PROMPT | llm | StrOutputParser()


def is_useful_doc(doc) -> bool:
    content = (doc.page_content or "").strip()
    chapter_id = str(doc.metadata.get("chapter_id", ""))

    # Filter out noisy records such as book-title-only vectors.
    if "." not in chapter_id:
        return False
    return len(content) >= MIN_CONTEXT_CHARS


def configure_console_encoding():
    if os.name != "nt":
        return

    # Ensure Korean input/output is preserved in Windows terminals.
    os.system("chcp 65001 > NUL")
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def rewrite_query(chain, question: str) -> str:
    rewritten = chain.invoke({"question": question}).strip()
    return rewritten or question


def retrieve_context(store, question: str, retrieval_query: str, k: int = RETRIEVAL_K):
    # Query with rewritten text first for multilingual reliability.
    primary_pairs = store.similarity_search_with_score(retrieval_query, k=k)
    useful_pairs = [(doc, score) for doc, score in primary_pairs if is_useful_doc(doc)]

    # Fallback to original query if rewritten query produced too few useful docs.
    if len(useful_pairs) < max(1, k // 2) and retrieval_query != question:
        fallback_pairs = store.similarity_search_with_score(question, k=k)
        useful_pairs = [(doc, score) for doc, score in fallback_pairs if is_useful_doc(doc)]

    docs = [doc for doc, _ in useful_pairs]
    return useful_pairs, docs


def ask(chain, question: str, context: str) -> str:
    return chain.invoke({"question": question, "context": context})


def has_false_insufficient_answer(answer: str) -> bool:
    lower = answer.lower()
    markers = [
        "cannot be found in the context",
        "not in the provided context",
        "context is insufficient",
        "insufficient context",
        "not enough context",
    ]
    return any(marker in lower for marker in markers)


def build_context_fallback_answer(docs) -> str:
    top_refs = [doc.metadata.get("reference", doc.metadata.get("chapter_id", "unknown")) for doc in docs[:3]]
    snippet = docs[0].page_content.strip().replace("\n", " ")[:220]
    return (
        "검색된 컨텍스트 기준으로 관련 구절이 있습니다. "
        f"참고: {', '.join(top_refs)}. "
        f"대표 구절 미리보기: {snippet}"
    )


def print_retrieval_query_debug(question: str, retrieval_query: str):
    if not RAG_DEBUG:
        return

    print("\n[DEBUG] Query")
    print(f"- user: {question}")
    print(f"- retrieval: {retrieval_query}")


def print_retrieval_debug(doc_score_pairs):
    if not RAG_DEBUG:
        return

    print("\n[DEBUG] Retrieval Results")
    print(f"- useful_count: {len(doc_score_pairs)}")
    for i, (doc, score) in enumerate(doc_score_pairs, start=1):
        reference = doc.metadata.get("reference", doc.metadata.get("chapter_id", "unknown"))
        print(f"- {i}. score={score:.4f}, ref={reference}, chars={len(doc.page_content)}")


def print_rejected_doc_debug(store, retrieval_query: str):
    if not RAG_DEBUG:
        return

    raw_pairs = store.similarity_search_with_score(retrieval_query, k=RETRIEVAL_K)
    rejected_pairs = [(doc, score) for doc, score in raw_pairs if not is_useful_doc(doc)]
    if not rejected_pairs:
        return

    print("[DEBUG] Rejected short/noisy docs")
    for i, (doc, score) in enumerate(rejected_pairs, start=1):
        reference = doc.metadata.get("reference", doc.metadata.get("chapter_id", "unknown"))
        print(f"- {i}. score={score:.4f}, ref={reference}, chars={len(doc.page_content)}")


if __name__ == "__main__":
    configure_console_encoding()

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )

    chain = build_rag_chain(llm)
    rewriter_chain = build_rewriter_chain(llm)

    print("=== Bible RAG Q&A (종료하려면 'exit' 입력) ===\n")
    print(
        f"Index: {PINECONE_INDEX}, Namespace: {PINECONE_NAMESPACE}, "
        f"k={RETRIEVAL_K}, min_chars={MIN_CONTEXT_CHARS}"
    )
    if RAG_DEBUG:
        print("[DEBUG] RAG_DEBUG=true (검색 결과를 출력합니다.)")

    while True:
        question = input("질문: ").strip()
        if question.lower() in ("exit", "quit", "종료"):
            print("종료합니다.")
            break
        if not question:
            continue

        retrieval_query = rewrite_query(rewriter_chain, question)
        print_retrieval_query_debug(question, retrieval_query)

        doc_score_pairs, docs = retrieve_context(vector_store, question, retrieval_query)
        print_retrieval_debug(doc_score_pairs)
        print_rejected_doc_debug(vector_store, retrieval_query)

        if not docs:
            print("\n답변: 검색된 컨텍스트가 없습니다. 인덱스/네임스페이스/데이터 적재 상태를 확인해주세요.\n")
            continue

        context = format_docs(docs)
        answer = ask(chain, question, context)
        if has_false_insufficient_answer(answer):
            answer = build_context_fallback_answer(docs)
        print(f"\n답변: {answer}\n")
