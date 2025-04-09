from retriever import load_retriever
from generator import load_generator, generate_answer
from SUPPORTED_MODELS import SUPPORTED_MODELS 

if __name__ == "__main__":
    retriever = load_retriever()
    model_key = "deepseek"

    generator = load_generator(model_key)
    # generator = load_generator(model_key="tinyllama")  # 可换成 "deepseek"

    print(f"Using model {SUPPORTED_MODELS[model_key]["hf_id"]}")

    while True:
        print("🧑‍🍳 RAG chef is ready. Ask your question!\n(Type 'exit' to quit)")
        query = input("🤔 You: ")
        if query.lower() in {"exit"}:
            break
        print("🤖 Generating...\n")
        docs = retriever.invoke(query)
        answer = generate_answer(query, docs, generator)
        print("🍽️ Answer:", answer)


# how to make a Chocolate Frosting
# give me a recipe of peppercorn cake

