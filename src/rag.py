import ollama
from retrieve import retrieve
from config import MODEL_NAME

def build_prompt(question, results, history):
    """
    Build a prompt using retrieved chunks as context.
    """

    documents = results["documents"][0]

    context = "\n\n".join(
        f"[Context {i + 1}]\n{document}"
        for i, document in enumerate(documents)
    )

    conversation = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in history
    )

    prompt = f"""
    You are answering questions about the book
    "AI Engineering: Building Applications With Foundation Models"
    by Chip Huyen.

    Use the provided context to answer the question.

    Rules:
    - Answer using the context whenever possible.
    - If the context is insufficient, say so.
    - Do not invent facts about the book.
    - Use conversation history to understand follow-up questions.
    - Conversation history is for context; retrieved passages are the
    primary evidence for answers about the book.
    - Give a clear, concise answer.

    Conversation history:
    {conversation if conversation else "No previous conversation."}

    Retrieved context:
    {context}

    Current question:
    {question}

    Answer:
    """
    return prompt

def generate_answer(prompt):
    """
    Stream the response from Ollama and return the complete answer.
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
    answer_parts = []

    for chunk in stream:
        text = chunk["message"]["content"]
        print(
            text,
            end="",
            flush=True,
        )
        answer_parts.append(text)
    return "".join(answer_parts)

def answer_question(question, history, top_k=5):
    """
    Retrieve relevant chunks and build the prompt.
    """

    results = retrieve(
        question,
        top_k=top_k
    )

    prompt = build_prompt(
        question,
        results,
        history,
    )

    return results, prompt


def print_sources(results):
    """Display the sources retrieved for the current question."""

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

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


def main():
    """
    Run a continuous question-and-answer session.
    """

    history = []

    print("=" * 70)
    print("AI ENGINEERING BOOK ASSISTANT")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 70)

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("\nChat ended. Goodbye!")
            break

        results, prompt = answer_question(
            question,
            history,
            top_k=5,
        )

        print("\n\nAssistant: ", end="", flush=True)

        answer = generate_answer(prompt)

        print_sources(results)

        # Save this exchange for future follow-up questions.
        history.append({
            "role": "user",
            "content": question,
        })
        history.append({
            "role": "assistant",
            "content": answer,
        })


if __name__ == "__main__":
    main()
