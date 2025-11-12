import gradio as gr
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import PyPDF2
import docx
import pandas as pd
import os
from typing import List, Dict

# Global state
embedder = None
db_client = None
collection = None
task_id = None
uploaded_files_info = []

# Initialize models
def load_embedder():
    """Load sentence transformer model"""
    global embedder
    if embedder is None:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return embedder

def get_vector_db():
    """Initialize ChromaDB"""
    global db_client
    if db_client is None:
        db_client = chromadb.Client(Settings(anonymized_telemetry=False))
    return db_client

def get_groq_client():
    """Initialize Groq client with API key"""
    api_key = os.getenv('GROQ_API_KEY', '')
    if not api_key:
        raise ValueError(" GROQ_API_KEY not found! Set it as an environment variable.")
    return Groq(api_key=api_key)

# Document processing
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_csv(file_path: str) -> str:
    """Extract text from CSV file"""
    df = pd.read_csv(file_path)
    return df.to_string()

def process_document(file_path: str) -> str:
    """Process uploaded file and extract text"""
    file_type = file_path.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            return extract_text_from_pdf(file_path)
        elif file_type == 'docx':
            return extract_text_from_docx(file_path)
        elif file_type == 'txt':
            return extract_text_from_txt(file_path)
        elif file_type == 'csv':
            return extract_text_from_csv(file_path)
        else:
            return ""
    except Exception as e:
        return f"Error processing file: {str(e)}"

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
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{file_name}_{i}" for i in range(len(chunks))]
    
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
    
    context_text = "\n\n".join([
        f"[Source: {doc['source']}]\n{doc['text']}" 
        for doc in context
    ])
    
    system_prompts = {
        "qa": "You are a helpful assistant that answers questions based on provided documents. Be concise and cite sources.",
        "research": "You are a research assistant that provides comprehensive analysis. Give detailed insights and connect ideas across sources.",
        "analysis": "You are a data analyst. Extract key metrics, patterns, and insights from the provided information."
    }
    
    system_prompt = system_prompts.get(task, system_prompts["qa"])
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# Gradio interface functions
def process_documents(files):
    """Process uploaded documents and return status"""
    global collection, uploaded_files_info
    
    if not files:
        return "Please upload at least one document.", gr.update(visible=False), gr.update(visible=False), "No documents loaded yet."
    
    try:
        db_client = get_vector_db()
        
        # Try to create collection, delete if it already exists
        try:
            collection = db_client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            # If collection exists, delete it and create a new one
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                try:
                    db_client.delete_collection(name="documents")
                except Exception:
                    pass
                # Create new collection after deletion
                collection = db_client.create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}
                )
            else:
                raise  # Re-raise if it's a different error
        
        uploaded_files_info = []
        for file in files:
            file_path = file.name
            file_name = os.path.basename(file_path)
            text = process_document(file_path)
            
            if text and not text.startswith("Error"):
                chunks = chunk_text(text)
                add_to_vector_db(chunks, file_name, collection)
                uploaded_files_info.append(file_name)
        
        if uploaded_files_info:
            status_msg = f"Successfully processed {len(uploaded_files_info)} document(s):\n"
            status_msg += "\n".join([f"  • {fname}" for fname in uploaded_files_info])
            status_msg += "\n\nSelect a task to continue."
            session_msg = f"{len(uploaded_files_info)} document(s) loaded"
            return status_msg, gr.update(visible=True), gr.update(visible=False), session_msg
        else:
            return "Error processing documents. Please try again.", gr.update(visible=False), gr.update(visible=False), "No documents loaded yet."
    except Exception as e:
        return f"Error: {str(e)}", gr.update(visible=False), gr.update(visible=False), "No documents loaded yet."

def select_task(task_type: str, current_task, session_info_text):
    """Handle task selection"""
    global task_id
    
    task_config = {
        "qa": {
            "id": "qa",
            "name": "Quick Q&A",
            "welcome": f"I've analyzed your {len(uploaded_files_info)} document(s). Ask me anything!"
        },
        "research": {
            "id": "research",
            "name": "Deep Research",
            "welcome": "Ready for deep research! I'll provide comprehensive analysis with connections across your documents."
        },
        "analysis": {
            "id": "analysis",
            "name": "Data Analysis",
            "welcome": "I'll help you analyze data from your documents. What metrics or patterns are you interested in?"
        }
    }
    
    config = task_config.get(task_type, task_config["qa"])
    task_id = config["id"]
    chat_history = [[None, config["welcome"]]]
    new_session_info = f" {len(uploaded_files_info)} document(s) loaded\nTask: **{config['name']}**"
    
    return gr.update(visible=True), gr.update(visible=True), chat_history, config["name"], config["name"], new_session_info

def chat_respond(message, history, current_task):
    """Handle chat messages"""
    global collection, task_id
    
    if not message or not collection:
        return history, current_task
    
    history.append([message, None])
    
    try:
        top_k = 5 if task_id == "research" else 3
        relevant_docs = query_documents(message, collection, top_k=top_k)
        response = generate_response(message, relevant_docs, task_id)
        
        sources_text = "\n\nSources:\n" + "\n".join([f"• {doc['source']} (Chunk {doc['chunk_id']})" for doc in relevant_docs])
        history[-1][1] = response + sources_text
    except Exception as e:
        history[-1][1] = f"Error: {str(e)}"
    
    return history, current_task

def reset_session(current_task, session_info_text):
    """Reset the session"""
    global collection, uploaded_files_info, task_id
    collection = None
    uploaded_files_info = []
    task_id = None
    return (
        gr.update(value=None),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        [],
        None,
        None,
        "No documents loaded yet."
    )

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="AI Document Assistant") as app:
    gr.Markdown("# AI Document Assistant\n### Intelligent RAG-powered chatbot for your documents")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Session Info")
            session_info = gr.Markdown("No documents loaded yet.")
            gr.Markdown("---")
            gr.Markdown("### Features")
            gr.Markdown("- Multi-format support (PDF, DOCX, CSV, TXT)\n- Task-specific responses\n- Source citations\n- Fast retrieval")
            gr.Markdown("---")
            reset_btn = gr.Button("New Session", variant="secondary")
        
        with gr.Column(scale=3):
            with gr.Group(visible=True) as upload_section:
                gr.Markdown("### Upload Your Documents")
                file_upload = gr.File(file_count="multiple", file_types=[".pdf", ".docx", ".txt", ".csv"], label="Choose files")
                process_btn = gr.Button("Process Documents", variant="primary")
                upload_status = gr.Markdown()
            
            with gr.Group(visible=False) as task_section:
                gr.Markdown("### Choose Your Task")
                with gr.Row():
                    task_qa = gr.Button("Quick Q&A\n\nFast answers to specific questions", scale=1)
                    task_research = gr.Button("Deep Research\n\nComprehensive analysis & insights", scale=1)
                    task_analysis = gr.Button("Data Analysis\n\nExtract & analyze structured data", scale=1)
            
            with gr.Group(visible=False) as chat_section:
                gr.Markdown("### Chat with Your Documents")
                chatbot = gr.Chatbot(label="Conversation", height=500, show_copy_button=True)
                task_display = gr.Markdown(visible=False)
                with gr.Row():
                    msg = gr.Textbox(label="Ask a question about your documents...", placeholder="Type your question here...", scale=4)
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
    
    current_task_state = gr.State(value=None)
    
    # Event handlers
    process_btn.click(
        fn=process_documents,
        inputs=[file_upload],
        outputs=[upload_status, task_section, chat_section, session_info]
    )
    
    def select_qa(ct, si):
        return select_task("qa", ct, si)
    
    def select_research(ct, si):
        return select_task("research", ct, si)
    
    def select_analysis(ct, si):
        return select_task("analysis", ct, si)
    
    task_qa.click(
        fn=select_qa,
        inputs=[current_task_state, session_info],
        outputs=[task_section, chat_section, chatbot, task_display, current_task_state, session_info]
    )
    
    task_research.click(
        fn=select_research,
        inputs=[current_task_state, session_info],
        outputs=[task_section, chat_section, chatbot, task_display, current_task_state, session_info]
    )
    
    task_analysis.click(
        fn=select_analysis,
        inputs=[current_task_state, session_info],
        outputs=[task_section, chat_section, chatbot, task_display, current_task_state, session_info]
    )
    
    submit_btn.click(
        fn=chat_respond,
        inputs=[msg, chatbot, current_task_state],
        outputs=[chatbot, current_task_state]
    ).then(fn=lambda: "", outputs=[msg])
    
    msg.submit(
        fn=chat_respond,
        inputs=[msg, chatbot, current_task_state],
        outputs=[chatbot, current_task_state]
    ).then(fn=lambda: "", outputs=[msg])
    
    reset_btn.click(
        fn=reset_session,
        inputs=[current_task_state, session_info],
        outputs=[file_upload, upload_section, task_section, chat_section, chatbot, task_display, current_task_state, session_info]
    ).then(fn=lambda: "", outputs=[upload_status])

if __name__ == "__main__":
    app.launch()

