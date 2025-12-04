from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import config

class RAGEngine:
    def __init__(self):
        print("🔧 Initializing RAGEngine...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
        
        try:
            self.collection = self.chroma_client.get_collection("contracts")
            print(f"✅ Found vector database with {self.collection.count()} chunks")
        except:
            print("❌ Vector database not found! Please run: python document_processor.py")
            raise
        
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        print("✅ Initialization complete")
        
    def retrieve(self, query: str, n_results: int = 3):
        """Retrieve relevant text chunks"""
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM"""
        prompt = f"""You are a bank contract analysis assistant. Please answer the user's question based on the following contract content.

Contract Content:
{context}

User Question: {query}

Requirements:
1. Use clear, plain language
2. If the contract has explicit information, reference it directly
3. If the contract doesn't contain relevant information, clearly tell the user
4. Keep the answer concise (under 150 words)
5. DO NOT use italics or special formatting in your response  
6. Use normal text with proper spacing

Answer:"""

        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional bank contract analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def answer_question(self, query: str):
        """Complete QA pipeline"""
        print(f"🔍 Retrieving relevant content...")
        
        # Retrieve
        results = self.retrieve(query, n_results=3)
        
        if not results['documents'][0]:
            return {
                "answer": "Sorry, I couldn't find relevant information in the contract.",
                "sources": []
            }
        
        # Combine retrieved text
        context = "\n\n".join(results['documents'][0])
        sources = results['metadatas'][0]
        
        print(f"📝 Generating answer...")
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": results['documents'][0]
        }

# Test code
if __name__ == "__main__":
    engine = RAGEngine()
    
    test_questions = [
        "What is the annual fee for this credit card?",
        "Will I be penalized for early repayment?",
        "What are the late payment fees?"
    ]
    
    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"❓ Question: {q}")
        result = engine.answer_question(q)
        print(f"💬 Answer: {result['answer']}")
        print(f"📄 Sources: {result['sources']}")