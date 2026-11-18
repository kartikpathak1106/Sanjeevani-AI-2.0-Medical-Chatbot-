"""
Lets a logged-in user upload their own medical PDF, which gets chunked,
embedded, and added into the existing FAISS vectorstore — so the RAG
chatbot can immediately start answering questions from it too.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

UPLOAD_DIR = "data/user_uploads"
DB_FAISS_PATH = "vectorstore/db_faiss"


def add_pdf_to_vectorstore(uploaded_file, vectorstore, embedding_model):
    """
    uploaded_file: a Streamlit UploadedFile object
    vectorstore: the currently loaded FAISS vectorstore (from get_vectorstore())
    Returns: number of chunks added
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(save_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore.add_documents(chunks)
    vectorstore.save_local(DB_FAISS_PATH)

    return len(chunks)
