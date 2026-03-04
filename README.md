# AI-Powered PDF Question Answering System

## Project Overview
This project implements an AI-powered Question Answering system using a Retrieval-Augmented Generation (RAG) pipeline.

The system allows users to upload a PDF document and ask natural language questions about its content. Instead of relying only on the language model’s internal knowledge, the system retrieves relevant document chunks from a vector database (ChromaDB) and generates grounded, context-aware answers using a Large Language Model (Groq / OpenRouter).

---

## System Architecture

The application follows a modular RAG architecture:

- Ingestion Layer: Extracts and splits text from PDF files using PyMuPDF.
- Embedding Layer: Converts text chunks into vector embeddings.
- Retrieval Layer: Stores and retrieves embeddings using ChromaDB (Similarity / MMR search).
- Generation Layer: Uses an LLM to generate grounded answers.
- UI Layer: Gradio-based web interface for document upload and chat interaction.

---

## Features

- Upload and process PDF documents
- Automatic text chunking
- Vector-based similarity retrieval
- Context-grounded answer generation
- Anti-hallucination guardrails
- Interactive Gradio web interface

---

## Technologies Used

- Python
- LangChain
- ChromaDB
- PyMuPDF
- Gradio
- Groq / OpenRouter API

---

## How to Run the Project

1. Create a virtual environment:
   py -3.10 -m venv venv

2. Activate it (Windows):
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Add your API key inside the .env file:
   OPENROUTER_API_KEY=your_key_here

5. Run the application:
   python app_ui.py

6. Open the generated local link in your browser.

---

## Project Structure

app/
│
├── ingestion/
├── retrieval/
├── llm/
├── rag_pipeline.py
│
test_data/
│   └── What is AI.pdf
│
app_ui.py
requirements.txt
README.md
EVALUATION_REPORT.md

---

## Test Data

A sample file (What is AI.pdf) is included inside the test_data folder for evaluation and demonstration purposes."# ai-pdf-qa-system" 
