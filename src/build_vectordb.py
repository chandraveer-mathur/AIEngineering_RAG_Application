import pymupdf
import chromadb

from sentence_transformers import SentenceTransformer
from tokenizer import load_tokenizer
from chunk import create_document_chunks
from config import PDF_PATH, EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME, TARGET_TOKENS
# --------------------------------------------------
# Configuration
# --------------------------------------------------

# --------------------------------------------------
# 1. Load tokenizer and embedding model
# --------------------------------------------------

print("Loading tokenizer...")
tokenizer = load_tokenizer()
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("Models loaded.")

# --------------------------------------------------
# 2. Open PDF
# --------------------------------------------------

print("Opening PDF...")
doc = pymupdf.open(PDF_PATH)

# --------------------------------------------------
# 3. Create chunks
# --------------------------------------------------

print("Creating chunks...")

chunks = create_document_chunks(
    doc,
    tokenizer,
    target_tokens=TARGET_TOKENS,
)

print("Total chunks:", len(chunks))


# --------------------------------------------------
# 4. Create ChromaDB client
# --------------------------------------------------

print("Opening ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

# --------------------------------------------------
# 5. Create collection
# --------------------------------------------------

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# --------------------------------------------------
# 6. Prepare text and metadata
# --------------------------------------------------

texts = [
    chunk["text"]
    for chunk in chunks
]

metadatas = [
    {
        "chapter": chunk["chapter"] or "",
        "section": chunk["section"] or "",
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "token_count": chunk["token_count"],
    }
    for chunk in chunks
]

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

# --------------------------------------------------
# 7. Create embeddings
# --------------------------------------------------

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
)

# --------------------------------------------------
# 8. Store everything in ChromaDB
# --------------------------------------------------

print("Adding chunks to ChromaDB...")

collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
)

# --------------------------------------------------
# 9. Verify database
# --------------------------------------------------

print("\n" + "=" * 60)
print("VECTOR DATABASE CREATED")
print("=" * 60)

print("Collection:", COLLECTION_NAME)
print("Stored chunks:", collection.count())
print("Database path:", CHROMA_PATH)