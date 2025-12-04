import streamlit as st
from rag_engine import RAGEngine
from document_processor import DocumentProcessor
import os
import tempfile

# Page config
st.set_page_config(
    page_title="Bank Contract QA Assistant",
    page_icon="💳",
    layout="wide"
)

# Title
st.title("💳 Bank Contract QA Assistant")
st.markdown("Upload a bank contract and ask questions in natural language!")

# Initialize session state
if 'engine' not in st.session_state:
    try:
        st.session_state.engine = RAGEngine()
        st.session_state.ready = True
    except:
        st.session_state.ready = False

# Sidebar - Upload contract
with st.sidebar:
    st.header("📄 Upload Contract")
    
    uploaded_file = st.file_uploader(
        "Choose a contract file",
        type=['txt', 'pdf'],
        help="Upload your bank contract (PDF or TXT)"
    )
    
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        if st.button("🔄 Process Contract"):
            with st.spinner("Processing contract..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Move to contracts directory
                contracts_dir = "data/contracts"
                os.makedirs(contracts_dir, exist_ok=True)
                new_path = os.path.join(contracts_dir, uploaded_file.name)
                
                with open(tmp_path, 'rb') as src:
                    with open(new_path, 'wb') as dst:
                        dst.write(src.read())
                
                os.unlink(tmp_path)
                
                # Process
                processor = DocumentProcessor()
                processor.process_contracts()
                
                # Reload engine
                st.session_state.engine = RAGEngine()
                st.session_state.ready = True
                
                st.success("✅ Contract processed successfully!")
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    if st.session_state.ready:
        try:
            count = st.session_state.engine.collection.count()
            st.success(f"✅ Ready ({count} chunks indexed)")
        except:
            st.error("❌ Vector database not found")
    else:
        st.warning("⚠️ Please process a contract first")

# Main area - QA
st.header("❓ Ask Questions")

if st.session_state.ready:
    # Example questions
    st.markdown("**Example questions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💵 What's the annual fee?"):
            st.session_state.question = "What is the annual fee for this credit card?"
    with col2:
        if st.button("⏰ Can I prepay?"):
            st.session_state.question = "Can I prepay my balance without penalty?"
    with col3:
        if st.button("⚠️ What are late fees?"):
            st.session_state.question = "What are the late payment fees?"
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        value=st.session_state.get('question', ''),
        placeholder="e.g., What is the APR for cash advances?"
    )
    
    if st.button("🔍 Get Answer", type="primary") and question:
        with st.spinner("Searching contract..."):
            result = st.session_state.engine.answer_question(question)
            
            # Display answer
            st.markdown("### 💬 Answer")
            st.info(result['answer'])
            
            # Display sources
            with st.expander("📄 View Sources"):
                st.markdown("**Retrieved contract sections:**")
                for i, chunk in enumerate(result['retrieved_chunks'], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.code(chunk, language="text")
                
                st.markdown("**Metadata:**")
                st.json(result['sources'])
else:
    st.warning("⚠️ Please upload and process a contract first (see sidebar)")
    
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit + RAG (Retrieval-Augmented Generation)</p>
    <p>Using OpenAI GPT-3.5 + Sentence Transformers + ChromaDB</p>
</div>
""", unsafe_allow_html=True)