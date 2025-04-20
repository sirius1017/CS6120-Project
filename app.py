# from retriever import load_retriever
# from generator import load_generator, generate_answer
from SUPPORTED_MODELS import SUPPORTED_MODELS 
from query_construction import query_classifier
from retriever import retrieve_full_recipes2, top_k_by_nutrient

def retrive_recipes(input:str):
    query = query_classifier(input)
    ids = retrieve_full_recipes2(query,"./recipes.json")
    return ids
    
def retrive_recipes_title(query:str):
    query["type"] = ["title"]
    ids = retrieve_full_recipes2(query,"./recipes.json")
    return ids

def retrive_recipes_ingredients(query:str):
    query["type"] = ["ingredients"]
    ids = retrieve_full_recipes2(query,"./recipes.json")
    return ids


def print_recipes(ids):
    def format_float(value):
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return value
        
    for i, r in enumerate(ids, start=1):
        title = r.get("title", "N/A")
        raw_ings = r.get("ingredients", [])
        ingredients = [i["text"].split(",")[0].lower() for i in raw_ings if "text" in i and i["text"]]
        
        nutrition = r.get("nutr_values_per100g", {})
        energy = format_float(nutrition.get("energy", "N/A"))
        fat = format_float(nutrition.get("fat", "N/A"))
        protein = format_float(nutrition.get("protein", "N/A"))
        salt = format_float(nutrition.get("salt", "N/A"))
        saturates = format_float(nutrition.get("saturates", "N/A"))
        sugars = format_float(nutrition.get("sugars", "N/A"))

        print(f"\n[{i}] {title}")
        print("Ingredients:", ", ".join(ingredients))
        print("Nutrition Facts (per 100g):")
        print(f"  Energy: {energy}")
        print(f"  Fat: {fat}")
        print(f"  Protein: {protein}")
        print(f"  Salt: {salt}")
        print(f"  Saturates: {saturates}")
        print(f"  Sugars: {sugars}")


if __name__ == "__main__":
    '''
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
        
        docs = retriever.invoke(query)

        answer = generate_answer(query, docs, generator)
        print("🍽️ Answer:", answer)
    '''

    print("🍳 Welcome to the Smart Recipe Finder!")
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Search recipes (general)")
        print("2. Search by title only")
        print("3. Search by ingredients only")
        print("0. Exit")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            user_query = input("Enter your recipe query: ")
            results = retrive_recipes(user_query)
            print_recipes(results)

        elif choice == "2":
            user_query = input("Enter title-related query: ")
            structured = query_classifier(user_query)
            results = retrive_recipes_title(structured)
            print_recipes(results)

        elif choice == "3":
            user_query = input("Enter ingredients-related query: ")
            structured = query_classifier(user_query)
            results = retrive_recipes_ingredients(structured)
            print_recipes(results)

        elif choice == "0":
            print("👋 Exiting. Enjoy your cooking!")
            break
        else:
            print("❌ Invalid option. Please choose 1-5.")
