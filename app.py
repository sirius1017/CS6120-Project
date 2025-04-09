from retriever import load_retriever
from generator import load_generator, generate_answer
from SUPPORTED_MODELS import SUPPORTED_MODELS 

if __name__ == "__main__":
    retriever = load_retriever()
    model_key = "deepseek"
    # model_key = "tinyllama"

    generator = load_generator(model_key)
    print(f'Using model {SUPPORTED_MODELS[model_key]["hf_id"]}')

    while True:
        print("🧑‍🍳 RAG chef is ready. Ask your question! (Type 'exit' to quit)")
        query = input("🤔 You: ")
        if query.lower() in {"exit"}:
            break
        print("🤖 Generating...\n")
        
        # 对query进行处理？
        docs = retriever.invoke(query)

        answer = generate_answer(query, docs, generator)
        print("🍽️ Answer:", answer)

