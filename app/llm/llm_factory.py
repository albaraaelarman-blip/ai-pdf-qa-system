import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# ✅ يتحمل مرة واحدة بس عند تشغيل البرنامج
load_dotenv(".env.groq")
load_dotenv(".env.openrouter")


def get_llm(provider: str):

    if provider == "Groq":

        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

    elif provider == "OpenRouter":

        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="meta-llama/llama-3.1-8b-instruct",
            temperature=0
        )

    else:
        raise ValueError("Unsupported provider")