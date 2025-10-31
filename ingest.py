import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define paths
DATA_PATH = "data"
DB_PATH = "db"

def create_vector_db():
    print("Loading documents...")
    
    # Loaders for different file types
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)

    # Load and combine documents
    pdf_documents = pdf_loader.load()
    txt_documents = txt_loader.load()
    documents = pdf_documents + txt_documents

    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} document(s).")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # Load embedding model
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Create and persist vector store
    print("Creating vector store...")
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=DB_PATH
    )
    print(f"Vector store created and saved at {DB_PATH}")

if __name__ == "__main__":
    create_vector_db()