import os
from dotenv import load_dotenv

# Load variable from .env
load_dotenv()


PDF_PATH = os.getenv("PDF_PATH")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "chroma_db",
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "ai_engineering",
)

TARGET_TOKENS = int(
    os.getenv(
        "TARGET_TOKENS",
        "400"
    )
)

if not PDF_PATH:
    raise ValueError(
        "PDF_PATH is missing from .env"
    )