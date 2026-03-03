# Evaluation Report: AI-Powered PDF Question Answering System

## 1. Project Overview

This report evaluates the performance and reliability of the implemented Retrieval-Augmented Generation (RAG) pipeline for document-based question answering.

The system integrates LangChain, ChromaDB, and high-performance LLMs (Groq/OpenRouter) to generate accurate, context-grounded responses from PDF documents.

---

## 2. Key Performance Metrics

The system was tested using the document:
What is AI.pdf

### Retrieval Accuracy
The system successfully retrieved specific facts, such as identifying John McCarthy as the person who coined the term "Artificial Intelligence" in 1956.

### Context Grounding (Anti-Hallucination)
When asked questions not present in the document (e.g., “Who is the current CEO of OpenAI?”), the system correctly responded:
"I don't know based on the document."

This demonstrates proper grounding and guardrail behavior.

### Latency
Average response time ranged between 2–5 seconds depending on document size and API latency, satisfying the non-functional requirement of <5 seconds.

### Synthesis Capability
The system effectively aggregated information across multiple sections of the document to generate coherent and comprehensive answers.

---

## 3. Technical Implementation Highlights

### Modular Architecture
The project is structured into independent ingestion, retrieval, and LLM modules, ensuring maintainability and scalability.

### Advanced Retrieval Strategy
Similarity search and MMR (Maximal Marginal Relevance) were used to retrieve diverse and relevant document chunks.

### Session Management
Gradio State and dynamic session handling ensure safe multi-user interactions without file conflicts.

---

## 4. Known Limitations

- Performance depends on the structure and quality of the PDF.
- Scanned PDFs without selectable text may require OCR support.
- Very large documents may increase latency.
- Extremely long summaries may exceed the LLM token limit.
- System performance depends on external API availability.

---

## 5. Conclusion

The AI-Powered PDF Question Answering System successfully integrates document retrieval with LLM-based generation.

It meets both functional and non-functional requirements by providing:
- Accurate document-grounded answers
- Controlled hallucination behavior
- Stable performance
- Clean modular architecture

The system demonstrates a practical and efficient implementation of a real-world RAG pipeline.