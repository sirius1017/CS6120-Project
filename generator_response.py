from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import retrieve_full_recipes
import os
from data_preprocessing import load_recipes_from_json
from query_construction import query_classifier
from ollama import chat, Client

load_dotenv()

def build_context_from_recipe_json(recipe):
    title = recipe.get("title", "Untitled")
    url = recipe.get("url", "")
    ingredients = recipe.get("ingredients", [])
    instructions = recipe.get("instructions", [])
    nutritions = recipe.get("nutr_values_per100g", [])

    # Format ingredients list
    ingredient_texts = [
        f"- {ing['text']}" for ing in ingredients if "text" in ing
    ]

    # Format instructions list
    instruction_texts = [
        f"- {step['text']}" for step in instructions if "text" in step
    ]

    # Format nutritions list
    nutrition_texts = [
        f"- {key}: {value:.2f} g/100g" for key, value in nutritions.items()
    ]

    context_parts = [
        f"Title: {title}",
        f"URL: {url}",
        "",
        "Ingredients:",
        "\n".join(ingredient_texts),
        "",
        "Instructions:",
        "\n".join(instruction_texts),
        "Nutrition (per 100g):",
        "\n".join(nutrition_texts)
    ]

    return "\n".join(context_parts)

# generator_cache = {}

# def load_generator(model_key="gemma3", device="cuda"):
#     """
#     Load a generator using the Ollama API.
#     model_key should match the Ollama model name, e.g., 'gemma:3b'
#     """
#     if model_key in generator_cache:
#         return generator_cache[model_key]

#     # You can customize the base_url if needed
#     client = Client(host='http://localhost:11434')  # default Ollama server

#     def ollama_generator(prompt):
#         response = client.chat(
#             model="gemma:3b",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response['message']['content']

#     generator_cache[model_key] = ollama_generator
#     return ollama_generator

def generate_response(query, path="./recipes.json"):


    gen_prompt = PromptTemplate.from_template("""
    You are a helpful recipe assistant. A user asked the following question:

    "{query}"

    Here are some potentially relevant recipes or fragments:

    {context}

    Some relevant recipes may not match the question. 
    Based on the question and the provided recipes, generate 1-2 recipes matching the user's requirements. 
    You need to pay attention to 
    -vague query: if the query is too general or open ended and there is no relavant context, 
        "I’m sorry, I couldn’t find a matching recipe based on your request." and tell user give some specific ideas
    -long query: try to analyze long query before generate response.
    -negation: if there are "without", "not" , "doesn't", but context may have, remember to replace and tell the user
    -Avoid hallucinating responses 
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
    
    also add source if there is a url
    """)

    query_classified = query_classifier(query)
    
    # Retrieving
    retrieved_docs = retrieve_full_recipes(query_classified, path)

    #Format recipes
    context = "\n\n".join([
        build_context_from_recipe_json(recipe) for recipe in retrieved_docs
    ])
    # if we use chatmodel api
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key = os.getenv("GOOGLE_API_KEY"))

    # llm = Client(host='http://localhost:11434')
    # llm_chain = gen_prompt | llm

    # response = llm_chain.invoke({"query": query, "context": context})
    # print(response.content)

    # Send to Ollama's Gemma model
    # Transfer the gen_prompt to string 
    prompt = gen_prompt.format(query=query, context=context)
    response = chat(
        model="gemma3:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    # Print the final answer
    print(response['message']['content'])
    
    return response['message']['content']
    
   

if __name__=="__main__":
    #query1 = "I've seen a recipe calls for cooking the milk, cream, and sugar until the sugar has dissolved. Then, we would mix with a cup, while adding vanilla extract. We need an ice cream maker for churning according to the manufacturer's directions, but I don't know if I have it. We would need to serve immediately or ripen in the freezer. Do you know what this recipe is for?"
    #generate_response(query1)

     query2 = "Can you find me a yogurt recipe without fruit"
     generate_response(query2)