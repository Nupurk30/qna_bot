import streamlit as st
import os
import warnings
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- Configuration ---
warnings.filterwarnings("ignore")

# --- Model Loading (Cached) ---
# We use @st.cache_resource to load these models only once
@st.cache_resource
def load_models():
    """
    Loads and caches the embedding model and the LLM.
    """
    print("Loading models...")
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load the local LLM
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        model_kwargs={"temperature": 0.1, "max_length": 512},
    )
    print("Models loaded successfully.")
    return embeddings, llm

# --- PDF Processing ---
def get_pdf_text(pdf_file):
    """
    Extracts text from an uploaded PDF file.
    """
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# --- Text Chunking ---
def get_text_chunks(text):
    """
    Splits the text into manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# --- Vector Store Creation ---
def get_vector_store(text_chunks, embeddings):
    """
    Creates a vector store from text chunks.
    This is temporary and in-memory.
    """
    if not text_chunks:
        return None
    vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

# --- RAG Chain Creation ---
def get_rag_chain(vector_store, llm):
    """
    Creates the RAG (Retrieval-Augmented Generation) chain.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    template = """
    Use this context to answer the question. If you don't know, say you don't know.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Chat with your PDF", page_icon="🤖")
    st.title("🤖 Chat with Your PDF")
    st.write("Upload a PDF file and ask questions about its content.")
    
    # --- Load Models ---
    # This will run only once
    with st.spinner("Loading AI models... This may take a minute."):
        embeddings, llm = load_models()
    
    # --- File Uploader ---
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    
    # --- Main Logic ---
    if uploaded_file is not None:
        # Check if this is a new file
        if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                # 1. Extract text
                raw_text = get_pdf_text(uploaded_file)
                
                # 2. Split into chunks
                text_chunks = get_text_chunks(raw_text)
                
                # 3. Create vector store
                vector_store = get_vector_store(text_chunks, embeddings)
                
                # 4. Create RAG chain and store in session
                st.session_state.rag_chain = get_rag_chain(vector_store, llm)
                st.session_state.file_name = uploaded_file.name
                st.success(f"Processed '{uploaded_file.name}' successfully!")
        
        # --- Question Input ---
        if "rag_chain" in st.session_state:
            user_question = st.text_input("Ask a question about your document:")
            
            if st.button("Get Answer"):
                if user_question:
                    with st.spinner("Thinking..."):
                        result = st.session_state.rag_chain.invoke(user_question)
                        st.subheader("Answer:")
                        st.write(result['result'])
                        
                        # (Optional) Show sources
                        st.subheader("Sources (from your doc):")
                        st.write(f"Found {len(result['source_documents'])} relevant chunks.")
                else:
                    st.warning("Please enter a question.")

if __name__ == "__main__":
    main()