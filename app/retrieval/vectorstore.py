import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


COLLECTION_NAME = "rag_collection"

load_dotenv(".env.openrouter")


def get_embeddings():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    return OpenAIEmbeddings(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="openai/text-embedding-3-small",
    )


def create_vector_store(chunks, db_path: str):
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=db_path,
        collection_name=COLLECTION_NAME,
    )


def load_vector_store(db_path: str):
    return Chroma(
        persist_directory=db_path,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )