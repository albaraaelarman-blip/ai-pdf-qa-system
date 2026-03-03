import os
import uuid
import gradio as gr

from app.ingestion.loader import load_and_split_pdf
from app.retrieval.vectorstore import create_vector_store
from app.rag_pipeline import answer_question


# -----------------------------
# PDF Processing
# -----------------------------
def process_pdf(file):
    if not file:
        return "⚠ Please upload a PDF file first.", False, ""

    try:
        session_id = uuid.uuid4().hex
        base_dir = "chroma_sessions"
        db_path = os.path.join(base_dir, f"db_{session_id}")

        os.makedirs(base_dir, exist_ok=True)

        chunks = load_and_split_pdf(file.name)
        create_vector_store(chunks, db_path)

        return "🟢 Document processed successfully.", True, db_path

    except Exception as e:
        return f"🔴 Error: {str(e)}", False, ""


# -----------------------------
# Chat Function
# -----------------------------
def chat_function(message, history, provider, is_vector_ready, db_path):
    if not is_vector_ready or not db_path:
        return "⚠ Please upload and process a document first."

    try:
        return answer_question(message, history, provider, db_path)
    except Exception as e:
        return f"🔴 Error: {str(e)}"


# -----------------------------
# UI Layout
# -----------------------------
with gr.Blocks(theme=gr.themes.Base(primary_hue="blue")) as demo:

    vector_ready_state = gr.State(False)
    db_path_state = gr.State("")

    gr.Markdown("## 🚀 Intelligent Document RAG System")
    gr.Markdown("Modular • Multi-Provider • Grounded Answers")

    with gr.Row():

        # Sidebar
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Document Setup")

            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process Document")
            status = gr.Textbox(label="System Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### ⚙️ Model Settings")

            provider_dropdown = gr.Dropdown(
                choices=["Groq", "OpenRouter"],
                value="Groq",
                label="LLM Provider",
            )

        # Chat Area
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")

            chatbot = gr.ChatInterface(
                fn=chat_function,
                additional_inputs=[
                    provider_dropdown,
                    vector_ready_state,
                    db_path_state,
                ],
            )

    process_btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=[status, vector_ready_state, db_path_state],
    )


demo.launch(css="footer {display:none;}")