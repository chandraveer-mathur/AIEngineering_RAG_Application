import chromadb

from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME
)

# --------------------------------------------------
# 1. Load embedding model
# --------------------------------------------------
print("Loading embedding model")

model = SentenceTransformer(EMBEDDING_MODEL)

# --------------------------------------------------
# 2. Open ChromaDB
# --------------------------------------------------

print("Opening ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

# --------------------------------------------------
# 3. Retrieval function
# --------------------------------------------------

def retrieve(
        query,
        top_k = 5
):
    """
    Retrieve the most relavant chunks for a query
    """

    # Convert query into an embedding
    query_embedding = model.encode([query])[0]

    # Search chromadb
    results = collection.query(
        query_embeddings= [
            query_embedding.tolist()
        ],
        n_results=top_k,
    )

    return results

# --------------------------------------------------
# 4. Test retrieval
# --------------------------------------------------

if __name__ == "__main__":
    query = input("\n Ask a question: ")

    results = retrieve(query, top_k=5)

    print("\n" + "=" * 70)
    print("RETRIEVED CHUNKS")
    print("=" * 70)
    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (document, metadata, distance) in enumerate(zip(
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):

        print("\n" + "-" * 70)

        print("Result:", i)
        print("Distance:", round(distance, 4))

        print(
            "Chapter:",
            metadata["chapter"]
        )

        print(
            "Section:",
            metadata["section"]
        )

        print(
            "Pages:",
            metadata["page_start"],
            "-",
            metadata["page_end"],
        )

        print(
            "Tokens:",
            metadata["token_count"]
        )

        print("\nText:")
        print(document)



