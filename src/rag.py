import ollama
from retrieve import retrieve
from config import MODEL_NAME

def build_prompt(question, results):
    """
    Build a prompt using retrieved chunks as context.
    """

    documents = results["documents"][0]

    context = "\n\n".join(
        f"[Context {i + 1}]\n{document}"
        for i, document in enumerate(documents)
    )

    prompt = f"""
    You are answering questions about the book
    "AI Engineering: Building Applications With Foundation Models"
    by Chip Huyen.

    Use the provided context to answer the question.

    Rules:
    - Answer using the context whenever possible.
    - If the context does not contain enough information, say so.
    - Do not invent facts.
    - Give a clear and concise explanation.

    Context:

    {context}

    Question:

    {question}

    Answer:
    """

    return prompt

def generate_answer(prompt):
    """
    Send the prompt to local ollama model
    """

    stream = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True,
    )

    for chunk in stream:
        text = chunk["message"]["content"]

        print(
            text,
            end="",
            flush=True,
        )

def answer_question(question, top_k=5):
    """
    Retriene relavant chunks and generate an answer
    """

    results = retrieve(
        question,
        top_k=top_k
    )

    prompt = build_prompt(
        question, 
        results
    )

    return results, prompt

if __name__ == "__main__":
    
    question = input("\nAsk a question:")

    results, prompt = answer_question(
        question,
        top_k=5,
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    # Stream answer directly to the terminal
    generate_answer(prompt)

    print("\n")

    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (metadata, distance) in enumerate(
        zip(metadatas, distances),
        start=1,
    ):

        print(
            f"{i}. "
            f"{metadata['chapter']} | "
            f"{metadata['section']} | "
            f"pages {metadata['page_start']}-"
            f"{metadata['page_end']} | "
            f"distance {distance:.4f}"
        )



