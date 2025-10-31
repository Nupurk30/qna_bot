import os
import warnings
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- NEW ---
# Import the new context manager
from contextlib import asynccontextmanager

# --- 1. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
warnings.filterwarnings("ignore")

# --- 2. CONFIGURATION ---
DB_PATH = "db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")

# --- 3. DATA MODEL ---
class Query(BaseModel):
    text: str

# --- 4. GLOBAL MODEL ---
# We will load this inside the lifespan function
qa_chain = None

# --- 5. LIFESPAN FUNCTION (THE NEW WAY) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the RAG chain on API startup.
    This runs only once when the server starts.
    """
    global qa_chain
    print("Loading RAG chain...")
    
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # Load the persistent vector store
    db = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )
    
    # Create the retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # Create the prompt template
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Helpful Answer:
    """
    prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )
    
    # Load the local LLM
    llm = HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL,
        task="text2text-generation",
        model_kwargs={"temperature": 0.1, "max_length": 512},
    )
    
    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print("RAG chain loaded successfully.")
    
    # This is required by the lifespan function
    yield
    
    # (You could add shutdown code here if you needed it)
    print("Shutting down...")


# --- 6. CREATE THE APP ---
# Tell FastAPI to use the new lifespan function
app = FastAPI(
    title="RAG API",
    description="An API for asking questions to a local RAG model.",
    lifespan=lifespan  # <-- THIS IS THE NEW PART
)

# --- 7. API ENDPOINTS ---
@app.post("/ask")
def ask_question(query: Query):
    """
    Answers a question using the loaded RAG chain.
    """
    if not qa_chain:
        return {"error": "RAG chain is not loaded yet."}, 503
        
    print(f"Received query: {query.text}")
    result = qa_chain.invoke(query.text)
    
    # Return the answer and the sources
    return {
        "answer": result['result'],
        "sources": [doc.metadata['source'] for doc in result['source_documents']]
    }

@app.get("/")
def read_root():
    """
    Root endpoint.
    """
    return {"message": "RAG API is running. Go to /docs for API documentation."}