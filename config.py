import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables or .env file")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
POSTGRES_URI = os.getenv("POSTGRES_URI")
USE_POSTGRES_CHECKPOINTER = os.getenv("USE_POSTGRES_CHECKPOINTER", "false").lower() == "true"
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4o")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SERVER_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")