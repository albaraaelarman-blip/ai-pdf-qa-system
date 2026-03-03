from app.retrieval.vectorstore import load_vector_store
from app.llm.llm_factory import get_llm

def answer_question(query: str, history: list, provider: str, db_path: str):
    # تمرير المسار الديناميكي
    vectorstore = load_vector_store(db_path)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 15
        }
    )

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    formatted_history = ""
    for item in history:
        # إذا كان الإصدار حديثاً ويرسل البيانات كـ Dictionary
        if isinstance(item, dict):
            role = "Human" if item.get("role") == "user" else "AI"
            formatted_history += f"{role}: {item.get('content')}\n"
            
        # إذا كان الإصدار يرسل البيانات كقائمة (List/Tuple)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            formatted_history += f"Human: {item[0]}\nAI: {item[1]}\n"
    llm = get_llm(provider)

    prompt = f"""
You are an expert, highly precise AI assistant. Your task is to answer the user's question based strictly on the provided context.

STRICT INSTRUCTIONS:
1. Answer directly and concisely. DO NOT use introductory phrases like "Based on the context," "Here is the answer," or "The context says."
2. If the user's query is incomplete, grammatically incorrect, or short (e.g., "what ai"), infer their intent and provide the answer directly without commenting on their grammar or the phrasing of the question.
3. Use ONLY the provided context. Do not use outside knowledge.
4. If the exact answer is not found in the context, reply EXACTLY with: "I don't know based on the document." Do not try to guess.

CHAT HISTORY:
{formatted_history}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = llm.invoke(prompt)
    return response.content