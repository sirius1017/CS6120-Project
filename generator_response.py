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
        f"- {key}: {value:.2f} /100g" for key, value in nutritions.items()
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

def extract_exclude_fields(query_dict):
    exclude_items = []

    for section in ["title", "ingredients", "instructions"]:
        if section in query_dict and "exclude" in query_dict[section]:
            exclude_items.extend(query_dict[section]["exclude"])

    return ", ".join(exclude_items)
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

def is_query_long(query: str, threshold: int = 25):
    """
    Selects the appropriate prompt template based on query length.
    
    Args:
        query (str): The user's input query.
        threshold (int): Word count threshold to consider a query "long".

    Returns:
        PromptTemplate: Either long_query_prompt or normal_query_prompt.
    """
    word_count = len(query.strip().split())
    if word_count > threshold:
        return True
    else:
        return False
    
def generate_response(query, path="./recipes.json"):
    query_classified = query_classifier(query)
    
    # Retrieving
    retrieved_docs = retrieve_full_recipes(query_classified, path)

    #Format recipes
    context = "\n\n".join([
        build_context_from_recipe_json(recipe) for recipe in retrieved_docs
    ])

    # Print retrieved documents
    print(context)
    exclude_items = extract_exclude_fields(query_classified)

    # Generate response based on different promot
    if len(retrieved_docs) == 0:
        # General Query
        gen_prompt = PromptTemplate.from_template("""
        You are a helpful recipe assistant. A user asked the following question:

        "{query}"

        The query is too general or open-ended, and no relevant recipes were found based on the current database.

        Please provide 2-3 general cooking ideas or themes that might inspire the user (e.g., “try a hearty soup,” “explore rice-based dishes,” “make a simple stir-fry”). Do not generate full recipes.

        Politely encourage the user to ask again with more specific details only for ingredients, such as:
        - Key ingredients they want to use or avoid

        The goal is to guide the user toward a clearer request without guessing or hallucinating.
        """)
    elif len(exclude_items) != 0:
        # negation based query
        gen_prompt = PromptTemplate.from_template("""
        You are a helpful recipe assistant. A user asked the following question:

        "{query}"

        Here are some potentially relevant recipes or fragments:

        {context}

        The user specifically requested to exclude the following ingredients or elements: {exclude_items}.
        You must strictly ensure none of these appear in your generated recipe, even if they appear in the context.

        Pay close attention to the following:

        1. **Exclusion / Negation Handling**:  
        - Remove or substitute any matching ingredients or instructions from the generated recipes.
        - Clearly acknowledge this in your response. Start the recipe with a statement like:  
            **"As requested, here is a recipe without {exclude_items}."**
        - Avoid including similar or related ingredients that may violate the intent of the exclusion (e.g., avoid oats if nuts are excluded unless clarified).

        2. **Relevance to User Intent**:  
        Ensure the recipe is aligned with other parts of the query (e.g., dish type, method) while respecting exclusions.

        The recipe should be practical, accurate, and follow sound cooking principles. Present it in this format:

        1. Recipe Title:
        [Create an appropriate title for the new recipe]

        2. Ingredients:
        [List all ingredients with precise measurements — none should include {exclude_items}]

        3. Cooking Instructions:
        [Provide clear, step-by-step cooking instructions]

        4. Nutritional Information (per 100g, kcal for energy, g for others):
        [Include estimated nutritional values per 100g]

        5. Cooking Tips:
        [Add helpful tips for best results]
        
        6. Inspiration Origin: 
        [Relevant recipe title]  
        """)
    elif is_query_long(query):
        # long query
        gen_prompt = PromptTemplate.from_template("""
        You are a helpful recipe assistant. A user asked the following detailed question:

        "{query}"

        Here are some potentially relevant recipes or fragments:

        {context}

        Some relevant recipes may not match the question.

        **Step-by-Step Reasoning Required**:  
        The query is long or detailed. Start by breaking down the key cooking-related elements, such as ingredients, cooking method, dish type, and dietary restrictions.  
        Think step by step to ensure accurate understanding. Disregard unrelated or vague parts of the query.

        Then generate 1-2 recipes that best match the cooking intent of the user.

        Avoid hallucinations and make sure the recipe is practical, accurate, and follows sound cooking principles.

        Format your answer like this:

        1. Recipe Title:
        [Appropriate title]

        2. Ingredients:
        [Precise measurements]

        3. Cooking Instructions:
        [Step-by-step instructions]

        4. Nutritional Information (per 100g, kcal for energy, g for others):
        [Include estimated nutritional values per 100g]

        5. Cooking Tips:
        [Helpful advice]

        6. Inspiration Origin: 
        [Relevant recipe title]
        """)
    else:
        # Normal query
        gen_prompt = PromptTemplate.from_template("""
        You are a helpful recipe assistant. A user asked the following question:

        "{query}"

        Here are some potentially relevant recipes or fragments:

        {context}

        Some relevant recipes may not match the question.
                                                   
        Based on the question and the provided recipes, generate 1-2 recipes that accurately match the user's requirements.                                                           
        
        Avoid hallucinations and make sure the recipe is practical, accurate, and follows sound cooking principles.

        Format your answer like this:

        1. Recipe Title:
        [Appropriate title]

        2. Ingredients:
        [Precise measurements]

        3. Cooking Instructions:
        [Step-by-step instructions]

        4. Nutritional Information (per 100g, kcal for energy, g for others):
        [Include estimated nutritional values per 100g]

        5. Cooking Tips:
        [Helpful advice]

        6. Inspiration Origin: 
        [Relevant recipe title]
        """)

    # if we use chatmodel api
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key = os.getenv("GOOGLE_API_KEY"))

    # # llm = Client(host='http://localhost:11434')
    # llm_chain = gen_prompt | llm

    # response = llm_chain.invoke({"query": query, "context": context, "exclude_items": exclude_items})
    # print(response.content)

    # Send to Ollama's Gemma model
    # Transfer the gen_prompt to string 
    prompt = gen_prompt.format(query=query, context=context, exclude_items=exclude_items)
    response = chat(
        model="gemma3:latest",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )

    # Print the final answer
    print(response['message']['content'])
    return(response['message']['content'])
    
   

if __name__=="__main__":
    # query1 = "I've seen a recipe calls for cooking the milk, cream, and sugar until the sugar has dissolved. Then, we would mix with a cup, while adding vanilla extract. We need an ice cream maker for churning according to the manufacturer's directions, but I don't know if I have it. We would need to serve immediately or ripen in the freezer. Do you know what this recipe is for?"
    # generate_response(query1)

    # query2 = "Can you find me a yogurt recipe without fruit"
    # generate_response(query2)

    # query3 = "give me a dessert without milk or nuts"
    # generate_response(query3)

    # query4 = "What to cook tonight?"
    # generate_response(query4)
    
    # query5 = "give me a recipe with mango and cream"
    # generate_response(query5)

    query6 = "I want to make yogurt"
    generate_response(query6)

    # query7 = "Give me a yogurt recipe"
    # generate_response(query7)

    # query8 = "What can I cook with no dairry n no meat?"
    # generate_response(query8)
