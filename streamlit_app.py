import streamlit as st
import os
import logging
import time
from typing import List, Tuple
import tempfile
import shutil

# Set up the page config
st.set_page_config(
    page_title="LocalGPT - Private Document Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
try:
    from utils import get_embeddings, check_system_compatibility, get_optimal_device, estimate_memory_usage
    from load_models import get_model_recommendations
    from run_localGPT import retrieval_qa_pipline
    from constants import SOURCE_DIRECTORY
    import torch
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents_ingested' not in st.session_state:
        st.session_state.documents_ingested = False
    if 'system_info' not in st.session_state:
        st.session_state.system_info = None

def display_system_info():
    """Display system compatibility information."""
    if st.session_state.system_info is None:
        st.session_state.system_info = check_system_compatibility()
    
    info = st.session_state.system_info
    
    with st.expander("🖥️ System Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Memory", f"{info['total_memory_gb']:.1f} GB")
            st.metric("Available Memory", f"{info['available_memory_gb']:.1f} GB")
            st.write(f"**Platform:** {info['platform']}")
            
        with col2:
            st.metric("MPS Available", "✅" if info['mps_available'] else "❌")
            st.metric("CUDA Available", "✅" if info['cuda_available'] else "❌")
            st.write(f"**Apple Silicon:** {'✅' if info['is_apple_silicon'] else '❌'}")

def display_model_recommendations():
    """Display model recommendations based on system memory."""
    if st.session_state.system_info:
        memory_gb = st.session_state.system_info['available_memory_gb']
        recommendations = get_model_recommendations(memory_gb)
        
        with st.expander("🤖 Recommended Models", expanded=True):
            st.write(f"**Based on {memory_gb:.1f}GB available memory:**")
            
            for model in recommendations['models']:
                if model['recommended']:
                    st.success(f"✅ **{model['name']}** - {model['size_gb']}GB - {model['performance']} Performance")
                else:
                    st.info(f"ℹ️ **{model['name']}** - {model['size_gb']}GB - {model['performance']} Performance")

def upload_documents():
    """Handle document upload and processing."""
    st.header("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'xlsx', 'csv', 'md'],
        help="Upload documents to chat with. Supported formats: PDF, TXT, DOCX, XLSX, CSV, MD"
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} files uploaded:**")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size} bytes)")
        
        if st.button("📥 Process Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save uploaded files to SOURCE_DOCUMENTS
                for i, file in enumerate(uploaded_files):
                    file_path = os.path.join(SOURCE_DIRECTORY, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                    
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Saved {file.name}")
                
                # Run ingestion
                status_text.text("Processing documents...")
                progress_bar.progress(0.8)
                
                # Import and run ingest
                import subprocess
                result = subprocess.run(['python', 'ingest.py', '--device_type', get_optimal_device()], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    progress_bar.progress(1.0)
                    status_text.text("✅ Documents processed successfully!")
                    st.session_state.documents_ingested = True
                    st.success("Documents have been processed and are ready for querying!")
                else:
                    st.error(f"Error processing documents: {result.stderr}")
                    
            except Exception as e:
                st.error(f"Error uploading documents: {str(e)}")

def initialize_qa_system():
    """Initialize the QA system."""
    if st.session_state.qa_system is None and st.session_state.documents_ingested:
        device_type = get_optimal_device()
        
        with st.spinner("🚀 Loading AI model... This may take a few minutes on first run."):
            try:
                st.session_state.qa_system = retrieval_qa_pipline(
                    device_type=device_type, 
                    use_history=False,
                    promptTemplate_type="llama3"
                )
                st.success("✅ AI model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading AI model: {str(e)}")
                logger.error(f"Error initializing QA system: {e}")

def chat_interface():
    """Main chat interface."""
    st.header("💬 Chat with Your Documents")
    
    if not st.session_state.documents_ingested:
        st.warning("⚠️ Please upload and process documents first!")
        return
    
    # Initialize QA system if needed
    initialize_qa_system()
    
    if st.session_state.qa_system is None:
        st.error("❌ AI model not loaded. Please check the logs.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}:** {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.qa_system(prompt)
                    answer = response['result']
                    sources = [doc.metadata.get('source', 'Unknown') for doc in response['source_documents']]
                    
                    st.markdown(answer)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Display sources
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sources, 1):
                            st.write(f"**Source {i}:** {source}")
                            
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def sidebar():
    """Create sidebar with controls and information."""
    with st.sidebar:
        st.title("🤖 LocalGPT")
        st.markdown("---")
        
        # Display system info
        display_system_info()
        
        # Display model recommendations
        display_model_recommendations()
        
        st.markdown("---")
        
        # Controls
        st.header("⚙️ Controls")
        
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🗑️ Clear Documents"):
            # Clear uploaded documents
            if os.path.exists(SOURCE_DIRECTORY):
                shutil.rmtree(SOURCE_DIRECTORY)
                os.makedirs(SOURCE_DIRECTORY, exist_ok=True)
            
            # Clear vector database
            if os.path.exists("./DB"):
                shutil.rmtree("./DB")
            
            st.session_state.documents_ingested = False
            st.session_state.qa_system = None
            st.session_state.messages = []
            st.success("✅ Documents and chat cleared!")
            st.rerun()
        
        st.markdown("---")
        
        # Information
        st.header("ℹ️ About")
        st.markdown("""
        **LocalGPT** allows you to chat with your documents privately. 
        Everything runs locally on your Mac M4 - no data leaves your computer!
        
        **Features:**
        - 🔒 100% Private
        - 🚀 Mac M4 Optimized  
        - 📄 Multiple file formats
        - 🤖 Powered by LLaMA
        """)

def main():
    """Main application function."""
    initialize_session_state()
    
    # Create sidebar
    sidebar()
    
    # Main content
    st.title("🤖 LocalGPT - Private Document Chat")
    st.markdown("Chat with your documents privately using AI. Optimized for Mac M4.")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📄 Upload Documents", "💬 Chat"])
    
    with tab1:
        upload_documents()
    
    with tab2:
        chat_interface()

if __name__ == "__main__":
    main() 