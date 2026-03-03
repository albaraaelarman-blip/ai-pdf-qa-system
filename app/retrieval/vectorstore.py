import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

COLLECTION_NAME = "rag_collection"

def get_embeddings():

    load_dotenv(".env.openrouter")
    

    return OpenAIEmbeddings(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),

        model="openai/text-embedding-3-small" 
    )

def create_vector_store(chunks, db_path: str):
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name=COLLECTION_NAME
    )

    return vectorstore

def load_vector_store(db_path: str):
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )