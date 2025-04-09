import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from SUPPORTED_MODELS import SUPPORTED_MODELS 


def ensure_model(model_key="tinyllama"):
    """确保模型存在本地并加载"""
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_key}'. Choose from {list(SUPPORTED_MODELS.keys())}.")

    model_info = SUPPORTED_MODELS[model_key]
    hf_id = model_info["hf_id"]
    cache_dir = model_info["cache_dir"]

    print(f"🔍 Checking for local model: {model_key} ...")
    
    # 如果缓存目录不存在，则触发下载
    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        print("⬇️ Downloading model from Hugging Face...")
    
    tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float32
    )

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
    )

    print(f"✅ Model '{model_key}' ready to use (device: {'cuda' if device == 0 else 'cpu'})")
    return pipe

def generate_response(model_pipe, user_input):
    """使用模型生成回应"""
    prompt = f"""You are a helpful cooking assistant.

Question: {user_input}
Answer:"""
    output = model_pipe(prompt)[0]["generated_text"]
    return output.replace(prompt, "").strip()


if __name__ == "__main__":
    # model_name = "tinyllama"
    model_key = "deepseek"

    pipe = ensure_model(model_key)
    print(f'Using model {SUPPORTED_MODELS[model_key]["hf_id"]}')
    
    while True:
        print("🧑‍🍳 No-RAG chef is ready. Type your cooking question!\n(Type 'exit' to quit)")
        print("🤔 You: ")
        user_input = input()
        if user_input.lower() in {"exit"}:
            break
        print("🤖 Generating...\n")
        response = generate_response(pipe, user_input)
        print("🍽️ Answer:", response)


