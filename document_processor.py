import os
import pdfplumber
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import config

class DocumentProcessor:
    def __init__(self):
        print("🔧 Initializing DocumentProcessor...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
        print("✅ Initialization complete")
        
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from file (supports PDF and TXT)"""
        if file_path.endswith('.pdf'):
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError("Only PDF and TXT files are supported")
    
    def chunk_text(self, text: str) -> List[dict]:
        """Split text into chunks"""
        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        chunk_id = 0
        
        for line in lines:
            if len(current_chunk) + len(line) < config.CHUNK_SIZE:
                current_chunk += line + "\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": current_chunk.strip(),
                        "chunk_id": chunk_id
                    })
                    chunk_id += 1
                current_chunk = line + "\n"
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": current_chunk.strip(),
                "chunk_id": chunk_id
            })
        
        return chunks
    
    def process_contracts(self, contracts_dir: str = "data/contracts"):
        """Process all contracts and store in vector database"""
        print("📄 Starting contract processing...")
        
        # Create or get collection
        try:
            collection = self.chroma_client.get_collection("contracts")
            print("📚 Using existing vector database")
        except:
            collection = self.chroma_client.create_collection("contracts")
            print("📚 Creating new vector database")
        
        # Process each file
        for filename in os.listdir(contracts_dir):
            file_path = os.path.join(contracts_dir, filename)
            print(f"  Processing: {filename}")
            
            # Extract text
            text = self.extract_text_from_file(file_path)
            
            # Chunk text
            chunks = self.chunk_text(text)
            print(f"  Generated {len(chunks)} text chunks")
            
            # Vectorize and store
            for chunk in chunks:
                embedding = self.embedding_model.encode(chunk["text"]).tolist()
                collection.add(
                    ids=[f"{filename}_{chunk['id']}"],
                    embeddings=[embedding],
                    documents=[chunk["text"]],
                    metadatas=[{"source": filename, "chunk_id": chunk["chunk_id"]}]
                )
        
        print("✅ All contracts processed successfully!")
        return collection

# Test code
if __name__ == "__main__":
    processor = DocumentProcessor()
    collection = processor.process_contracts()
    print(f"📊 Vector database contains {collection.count()} text chunks")