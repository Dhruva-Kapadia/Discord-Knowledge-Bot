import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Discord Configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

# Model Configuration
# Using Google Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Model Names
# Gemini 2.5 Flash
# Note: Ensure the exact model string is correct for the API. 
# Common format: 'gemini-1.5-flash' or similar if 2.5 isn't public yet, 
# but complying with user request for "gemini-2.5-flash".
MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-2.5-flash') 
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'models/text-embedding-004')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHROMA_DB_DIR = os.path.join(DATA_DIR, 'chroma_db')
