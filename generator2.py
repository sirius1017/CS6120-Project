from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import retrieve_full_recipes
import os
from data_preprocessing import load_recipes_from_json
from query_construction import query_classifier

load_dotenv()
def extract_ids_from_retrieved_docs(nested_retrieved_docs):
    ids = set()
    for doc_group in nested_retrieved_docs:
        for doc, _score in doc_group:
            recipe_id = doc.metadata.get("id")
            if recipe_id:
                ids.add(recipe_id)
    return list(ids)

def get_recipes_by_ids(recipe_ids, full_recipes_json):
    id_set = set(recipe_ids)
    return [recipe for recipe in full_recipes_json if recipe.get("id") in id_set]

def build_context_from_recipe_json(recipe):
    title = recipe.get("title", "Untitled")
    url = recipe.get("url", "")
    ingredients = recipe.get("ingredients", [])
    instructions = recipe.get("instructions", [])
    nutritions = 

    # Format ingredients list
    ingredient_texts = [
        f"- {ing['text']}" for ing in ingredients if "text" in ing
    ]

    # Format instructions list
    instruction_texts = [
        f"- {step['text']}" for step in instructions if "text" in step
    ]

    # Format nutritions list
    instruction_texts = [
        f"- {step['text']}" for step in instructions if "text" in step
    ]

    context_parts = [
        f"Title: {title}",
        f"URL: {url}",
        "",
        "Ingredients:",
        "\n".join(ingredient_texts),
        "",
        "Instructions:",
        "\n".join(instruction_texts)
    ]

    return "\n".join(context_parts)

def generate_response(query, path="./recipes.json"):
    gen_prompt = PromptTemplate.from_template("""
    You are a helpful recipe assistant. A user asked the following question:

    "{query}"

    Here are some potentially relevant recipes or fragments:

    {context}

    Some relevant recipes may not match the question. 
    Based on the question and the provided recipes, generate 1-2 recipes matching the user's requirements. 
    The recipe should be practical, accurate, balanced, and follow proper cooking principles. Present it in this format:

    1. Recipe Title:
    [Create an appropriate title for the new recipe]

    2. Ingredients:
    [List all ingredients with precise measurements]

    3. Cooking Instructions:
    [Provide clear, step-by-step cooking instructions]

    4. Nutritional Information:
    [Include estimated nutritional values per 100g]

    5. Cooking Tips:
    [Add helpful tips for best results]
    """)

    query = "Give me a recipe containing only egg, milk, flour."
    query_classified = query_classifier(query)
    
    # Retrieving
    retrieved_docs = retrieve_full_recipes(query_classified, path)
    # Extract recipe id from retrieved documents and find the original documents
    retrieved_ids = extract_ids_from_retrieved_docs(retrieved_docs)
    full_recipes = load_recipes_from_json(path)
    matched_recipes = get_recipes_by_ids(retrieved_ids, full_recipes)

    # if we use chatmodel api
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key = os.getenv("GOOGLE_API_KEY"))
    llm_chain = gen_prompt | llm
    context = "\n\n".join([
        build_context_from_recipe_json(recipe) for recipe in matched_recipes
    ])
    response = llm_chain.invoke({"query": query, "context": context})
    print(response)