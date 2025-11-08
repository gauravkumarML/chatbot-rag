import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import Pypdf2
import docx
import pandas as pd
import io
import hashlib
from typing import List, Dict
import os

# Page config
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: 600;
        border-radius: 8px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: slideIn 0.3s ease-out;
    }
    .user-message {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    @keyframes slideIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 'upload'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'task_selected' not in st.session_state:
    st.session_state.task_selected = None

# Initialize models (cached for performance)
@st.cache_resource
def load_embedder():
    """Load sentence transformer model (runs locally, free)"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_vector_db():
    """Initialize ChromaDB (in-memory for demo)"""
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    return client

def get_groq_client():
    """Initialize Groq client with API key"""
    api_key = os.getenv('GROQ_API_KEY', st.secrets.get('GROQ_API_KEY', ''))
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found! Add it to Streamlit secrets.")
        st.stop()
    return Groq(api_key=api_key)

# Document processing functions
def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    reader = pypdf2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file) -> str:
    """Extract text from DOCX file"""
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file) -> str:
    """Extract text from TXT file"""
    return file.read().decode('utf-8')

def extract_text_from_csv(file) -> str:
    """Extract text from CSV file"""
    df = pd.read_csv(file)
    return df.to_string()

def process_document(file) -> str:
    """Process uploaded file and extract text"""
    file_type = file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            return extract_text_from_pdf(file)
        elif file_type == 'docx':
            return extract_text_from_docx(file)
        elif file_type == 'txt':
            return extract_text_from_txt(file)
        elif file_type == 'csv':
            return extract_text_from_csv(file)
        else:
            return ""
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def add_to_vector_db(chunks: List[str], file_name: str, collection):
    """Add document chunks to vector database"""
    embedder = load_embedder()
    
    # Generate embeddings
    embeddings = embedder.encode(chunks).tolist()
    
    # Create unique IDs
    ids = [f"{file_name}_{i}" for i in range(len(chunks))]
    
    # Add to collection
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids,
        metadatas=[{"source": file_name, "chunk_id": i} for i in range(len(chunks))]
    )

def query_documents(question: str, collection, top_k: int = 3) -> List[Dict]:
    """Query vector database for relevant chunks"""
    embedder = load_embedder()
    query_embedding = embedder.encode([question]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    return [
        {
            "text": doc,
            "source": meta["source"],
            "chunk_id": meta["chunk_id"]
        }
        for doc, meta in zip(results['documents'][0], results['metadatas'][0])
    ]

def generate_response(question: str, context: List[Dict], task: str) -> str:
    """Generate response using Groq API"""
    client = get_groq_client()
    
    # Build context string
    context_text = "\n\n".join([
        f"[Source: {doc['source']}]\n{doc['text']}" 
        for doc in context
    ])
    
    # Task-specific system prompts
    system_prompts = {
        "qa": "You are a helpful assistant that answers questions based on provided documents. Be concise and cite sources.",
        "research": "You are a research assistant that provides comprehensive analysis. Give detailed insights and connect ideas across sources.",
        "analysis": "You are a data analyst. Extract key metrics, patterns, and insights from the provided information."
    }
    
    system_prompt = system_prompts.get(task, system_prompts["qa"])
    
    # Generate response
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",  # Fast and high quality
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# Main app
def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: white;'>🧠 AI Document Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7);'>Intelligent RAG-powered chatbot for your documents</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Session Info")
        
        if st.session_state.documents_processed:
            st.success(f"✅ {len(st.session_state.uploaded_files)} document(s) loaded")
            if st.session_state.task_selected:
                st.info(f"🎯 Task: **{st.session_state.task_selected}**")
        
        st.markdown("---")
        st.markdown("### 🚀 Features")
        st.markdown("- Multi-format support (PDF, DOCX, CSV, TXT)")
        st.markdown("- Task-specific responses")
        st.markdown("- Source citations")
        st.markdown("- Fast retrieval (Groq API)")
        
        st.markdown("---")
        if st.button("🔄 New Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Step 1: Upload Documents
    if st.session_state.step == 'upload':
        col1, col2, col3 = st.columns([1,2,1])
        
        with col2:
            st.markdown("### 📤 Upload Your Documents")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'docx', 'txt', 'csv'],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} file(s) selected")
                for file in uploaded_files:
                    st.text(f"📄 {file.name} ({file.size/1024:.1f} KB)")
                
                if st.button("🚀 Process Documents", type="primary"):
                    with st.spinner("Processing documents..."):
                        # Initialize vector DB
                        db_client = get_vector_db()
                        collection = db_client.create_collection(
                            name="documents",
                            metadata={"hnsw:space": "cosine"}
                        )
                        
                        # Process each file
                        progress_bar = st.progress(0)
                        for idx, file in enumerate(uploaded_files):
                            text = process_document(file)
                            if text:
                                chunks = chunk_text(text)
                                add_to_vector_db(chunks, file.name, collection)
                            progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                        st.session_state.collection = collection
                        st.session_state.uploaded_files = uploaded_files
                        st.session_state.documents_processed = True
                        st.session_state.step = 'task'
                        st.rerun()
    
    # Step 2: Select Task
    elif st.session_state.step == 'task':
        st.markdown("### 🎯 Choose Your Task")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💬 Quick Q&A\n\nFast answers to specific questions", use_container_width=True):
                st.session_state.task_selected = "Quick Q&A"
                st.session_state.task_id = "qa"
                st.session_state.step = 'chat'
                st.session_state.messages = [
                    {"role": "assistant", "content": f"I've analyzed your {len(st.session_state.uploaded_files)} document(s). Ask me anything!"}
                ]
                st.rerun()
        
        with col2:
            if st.button("🧠 Deep Research\n\nComprehensive analysis & insights", use_container_width=True):
                st.session_state.task_selected = "Deep Research"
                st.session_state.task_id = "research"
                st.session_state.step = 'chat'
                st.session_state.messages = [
                    {"role": "assistant", "content": "Ready for deep research! I'll provide comprehensive analysis with connections across your documents."}
                ]
                st.rerun()
        
        with col3:
            if st.button("📊 Data Analysis\n\nExtract & analyze structured data", use_container_width=True):
                st.session_state.task_selected = "Data Analysis"
                st.session_state.task_id = "analysis"
                st.session_state.step = 'chat'
                st.session_state.messages = [
                    {"role": "assistant", "content": "I'll help you analyze data from your documents. What metrics or patterns are you interested in?"}
                ]
                st.rerun()
    
    # Step 3: Chat Interface
    elif st.session_state.step == 'chat':
        # Display chat messages
        for message in st.session_state.messages:
            css_class = "user-message" if message["role"] == "user" else "assistant-message"
            st.markdown(f'<div class="chat-message {css_class}">{message["content"]}</div>', unsafe_allow_html=True)
            
            # Show sources if available
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for source in message["sources"]:
                        st.text(f"• {source['source']} (Chunk {source['chunk_id']})")
        
        # Chat input
        user_input = st.chat_input("Ask a question about your documents...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Thinking..."):
                # Query vector DB
                relevant_docs = query_documents(
                    user_input, 
                    st.session_state.collection,
                    top_k=5 if st.session_state.task_id == "research" else 3
                )
                
                # Generate response
                response = generate_response(
                    user_input,
                    relevant_docs,
                    st.session_state.task_id
                )
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": relevant_docs
                })
            
            st.rerun()

if __name__ == "__main__":
    main()
