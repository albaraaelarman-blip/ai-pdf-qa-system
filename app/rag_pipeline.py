from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.retrieval.vectorstore import load_vector_store
from app.llm.llm_factory import get_llm


def answer_question(query: str, history: list, provider: str, db_path: str):
    vectorstore = load_vector_store(db_path)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 15},
    )

    llm = get_llm(provider)

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert, highly precise AI assistant.
Answer strictly based on the provided context.

RULES:
- Answer directly and concisely.
- Use ONLY the provided context.
- If the answer is not found, reply EXACTLY with:
"I don't know based on the document."

CHAT HISTORY:
{history}

CONTEXT:
{context}

QUESTION:
{question}
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    formatted_history = format_history(history)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "history": lambda _: formatted_history,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)


def format_history(history: list) -> str:
    formatted = []

    for item in history:
        if isinstance(item, dict):
            role = "Human" if item.get("role") == "user" else "AI"
            formatted.append(f"{role}: {item.get('content')}")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            formatted.append(f"Human: {item[0]}")
            formatted.append(f"AI: {item[1]}")

    return "\n".join(formatted)