# generator.py

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os
from SUPPORTED_MODELS import SUPPORTED_MODELS 


# 用于避免重复加载的缓存
generator_cache = {}

def load_generator(model_key="deepseek", device="cpu"):
    """
    Load a text generation pipeline for the specified model.
    model_key: one of ['deepseek', 'tinyllama']
    """
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model key: {model_key}. Use one of {list(SUPPORTED_MODELS.keys())}")
    
    if model_key in generator_cache:
        return generator_cache[model_key]

    model_name = SUPPORTED_MODELS[model_key]["hf_id"]
    cache_dir = SUPPORTED_MODELS[model_key]["cache_dir"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float32
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" and torch.cuda.is_available() else -1,
        max_new_tokens=100, ###############可调整
        do_sample=True,
        temperature=0.7,
    )

    generator_cache[model_key] = pipe
    return pipe

def generate_answer(query: str, context_docs: list, generator_pipeline):
    """Format prompt with context and generate answer using the generator."""
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = f"""You are a helpful cooking assistant. Use the following recipes as context to answer the user's question.

Context:
{context}

Question: {query}

Answer:"""

    response = generator_pipeline(prompt)[0]["generated_text"]
    return response.strip()
