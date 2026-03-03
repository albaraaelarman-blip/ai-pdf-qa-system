from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path: str):
    # استخدام PyMuPDF لأنه أسرع وأدق بكثير في استخراج النصوص
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # نحتفظ بنفس إعدادات التقطيع الممتازة الخاصة بك
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,      
        chunk_overlap=300     
    )

    chunks = splitter.split_documents(documents)
    return chunks