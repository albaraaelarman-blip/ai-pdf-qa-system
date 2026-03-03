import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


load_dotenv(".env.groq")
load_dotenv(".env.openrouter")


def get_llm(provider: str):
    provider = provider.strip()

    if provider == "Groq":
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
        )

    if provider == "OpenRouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model="meta-llama/llama-3.1-8b-instruct",
            temperature=0,
        )

    raise ValueError(f"Unsupported provider: {provider}")