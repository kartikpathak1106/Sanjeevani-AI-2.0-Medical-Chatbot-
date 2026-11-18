from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
import torch

load_dotenv(find_dotenv())


DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def get_embedding_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Embedding Device:", device)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": device
        },
        encode_kwargs={
            "batch_size": 64
        },
        show_progress=True
    )

    return embedding_model


print("Loading PDF files...")

documents = load_pdf_files(DATA_PATH)

print("Length of PDF pages:", len(documents))


print("Creating text chunks...")

text_chunks = create_chunks(documents)

print("Length of Text Chunks:", len(text_chunks))


print("Loading embedding model...")

embedding_model = get_embedding_model()


print("\nCreating FAISS vector database...")
print("Please wait. Embedding 23,000+ chunks can take time on CPU.\n")


db = FAISS.from_documents(
    text_chunks,
    embedding_model
)


db.save_local(DB_FAISS_PATH)


print("\n======================================")
print("FAISS DATABASE CREATED SUCCESSFULLY")
print("======================================")
print("Location:", DB_FAISS_PATH)
print("Total chunks:", len(text_chunks))