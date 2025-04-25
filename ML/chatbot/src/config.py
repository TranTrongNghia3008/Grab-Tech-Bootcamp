import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Database Configuration ---
DB_USER = os.getenv("PG_USER")
DB_PASSWORD = os.getenv("PG_PASSWORD")
DB_HOST = os.getenv("PG_HOST")
DB_PORT = os.getenv("PG_PORT", "5432")
DB_NAME = os.getenv("PG_DATABASE")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    raise ValueError("Missing one or more PostgreSQL connection environment variables (PG_USER, PG_PASSWORD, PG_HOST, PG_DATABASE)")

DB_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- API Keys ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable")

# --- Directory Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent # Points to the root 'chatbot_project' dir
PLOT_DIR = BASE_DIR / "plots"
LOCAL_CHART_DIR = BASE_DIR / "local_chart_images"

# Create directories if they don't exist
PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CHART_DIR.mkdir(parents=True, exist_ok=True)

# --- Agent Configuration ---
AGENT_MODEL = "gpt-4o-mini" # Or your preferred model
AGENT_TEMPERATURE = 0.0
MAX_SAMPLE_ROWS = 100 # Max rows for get_sample_data
DISTRIBUTION_TOP_N = 20 # Default N for value distribution

# --- Visualization Defaults ---
DEFAULT_FIGSIZE = (10, 6)
PIE_CHART_MAX_SLICES = 15