import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from SUPPORTED_MODELS import SUPPORTED_MODELS 
from query_construction import query_classifier
from retriever import retrieve_full_recipes2


generator_cache = {}

def load_generator(model_key="deepseek", device="cuda"):
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
        max_new_tokens=500,  # Increased to allow for longer responses
        do_sample=True,
        temperature=0.7,
        top_p=0.9,  # Added top_p sampling
        repetition_penalty=1.2  # Added repetition penalty
    )

    generator_cache[model_key] = pipe
    return pipe

def format_recipe(recipe):
    """Format a recipe into a readable string."""
    title = recipe.get("title", "Untitled Recipe")
    ingredients = [ing["text"] for ing in recipe.get("ingredients", []) if "text" in ing]
    instructions = recipe.get("instructions", "No instructions available")
    nutrition = recipe.get("nutr_values_per100g", {})
    
    return f"""Recipe: {title}
Ingredients: {', '.join(ingredients)}
Instructions: {instructions}
Nutrition (per 100g): {', '.join(f'{k}: {v}' for k, v in nutrition.items())}"""

def generate_answer(query: str, generator_pipeline, json_path: str = "./recipes.json"):
    """Format prompt with context and generate answer using the generator."""
    # Get query classification
    query_info = query_classifier(query)
    
    # Retrieve relevant recipes using the retriever
    retrieved_recipes = retrieve_full_recipes2(
        query=query_info,
        json_path=json_path,
        top_k=10
    )
    
    # Format the classification information
    classification_info = f"""Query Analysis:
- Intent: {query_info['intent']}
- Types: {', '.join(query_info['type'])}"""
    
    # Add specific details if present
    if query_info['ingredients']['include']:
        classification_info += f"\n- Required Ingredients: {query_info['ingredients']['include']}"
    if query_info['ingredients']['exclude']:
        classification_info += f"\n- Excluded Ingredients: {query_info['ingredients']['exclude']}"
    if query_info['cooking_methods']['include']:
        classification_info += f"\n- Cooking Methods: {query_info['cooking_methods']['include']}"
    if query_info['nutritions']:
        classification_info += f"\n- Nutrition Focus: {query_info['nutritions']} ({'high' if query_info['descending'] else 'low'})"

    # Format retrieved recipes
    context = "\n\n".join(format_recipe(recipe) for recipe in retrieved_recipes)
    
    prompt = f"""You are a helpful cooking assistant. Select the most appropriate recipe from the available recipes below and present it in a clear format.

{classification_info}

Available Recipes:
{context}

Question: {query}

Select one recipe from the available recipes above and present it in this exact format:

1. Recipe Title:
[Copy the exact title from the selected recipe]

2. Ingredients:
[List the ingredients exactly as they appear in the recipe]

3. Cooking Instructions:
[Copy the cooking instructions exactly as they appear in the recipe]

4. Nutritional Information:
[Copy the nutritional information exactly as it appears in the recipe]

Answer: Here is a recipe that matches your requirements:"""

    response = generator_pipeline(prompt)[0]["generated_text"]
    # Extract only the answer part after "Answer:"
    answer = response.split("Answer:")[-1].strip()
    return answer
