import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ Please set OPENAI_API_KEY in your .env file!")

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"

# Vector database configuration
VECTOR_DB_PATH = "./vector_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Legal compliance limits
LEGAL_LIMITS = {
    "max_apr": 27.375,  # Maximum legal APR (%)
    "max_late_fee_rate": 5.0,  # Maximum late fee rate (%)
}

print("✅ Configuration loaded successfully!")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")