import ollama

MODEL_NAME = "qwen2.5:3b"

def generate(prompt):
    response = ollama.chat(
        model = MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]

if __name__ == "__main__":
    prompt = input("Ask me -")
    answer = generate(prompt)
    print("ANSWER \n")
    print(answer)