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
    try:
        # إنشاء اسم مجلد فريد لهذه الجلسة
        session_id = uuid.uuid4().hex
        db_path = f"chroma_sessions/db_{session_id}"
        
        # التأكد من وجود المجلد الرئيسي
        os.makedirs("chroma_sessions", exist_ok=True)

        chunks = load_and_split_pdf(file.name)
        # نمرر المسار الجديد
        create_vector_store(chunks, db_path)

        # نرجع: رسالة النجاح، حالة الجاهزية (True)، ومسار قاعدة البيانات
        return "🟢 Document processed successfully.", True, db_path

    except Exception as e:
        # في حال الخطأ نرجع: رسالة الخطأ، الجاهزية (False)، ومسار فارغ
        return f"🔴 Error: {str(e)}", False, ""


# -----------------------------
# Chat Function
# -----------------------------
def chat_function(message, history, provider, is_vector_ready, db_path):
    if not is_vector_ready or not db_path:
        return "⚠ Please upload and process a document first."

    try:
        # نمرر db_path للدالة
        return answer_question(message, history, provider, db_path)
    except Exception as e:
        return f"Error: {str(e)}"


# -----------------------------
# UI Layout
# -----------------------------
with gr.Blocks() as demo:
    
    vector_ready_state = gr.State(False)
    # State جديد لحفظ مسار قاعدة البيانات الخاص بالمستخدم
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
                label="LLM Provider"
            )

        # Chat Area
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")

            chatbot = gr.ChatInterface(
                fn=chat_function,
                additional_inputs=[
                    provider_dropdown, 
                    vector_ready_state, 
                    db_path_state # تمرير الـ State للمحادثة
                ]
            )

    # عند المعالجة، نحدث 3 أشياء: النص، حالة الجاهزية، ومسار الـ DB
    process_btn.click(
        fn=process_pdf, 
        inputs=pdf_input, 
        outputs=[status, vector_ready_state, db_path_state]
    )

demo.launch(
    theme=gr.themes.Base(primary_hue="blue"),
    css="footer {display:none;}"
)