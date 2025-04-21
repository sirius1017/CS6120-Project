from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os, json, re

load_dotenv()

def query_classifier(query):
    classifier_prompt = PromptTemplate.from_template(
        """You are a helpful AI cooking assistant. Your job is to analyze a user's recipe-related query and extract structured information for recipe retrieval.
        1. Check the query is open-end or specific:
        - `"intent": "open_ended"` if the user query is vague or general, such as "What should I cook tonight?" or "Show me something good"
        - `"intent": "specific"` if the query includes details like ingredients, cooking methods, or dish types
        
        2. Classify the user query into following categories (can be multiple):
        - ingredients: mentions specific ingredients (e.g., "chicken and garlic")
        - title: resembles a recipe name or dish (e.g., "how to make ramen")
        - cooking_method: mentions a cooking technique (e.g., "boiling" or "baking")

        3. Extract key ingredients or concepts (as a list of keywords) from the query to assist in retrieving recipes. 
        - For ingredients_query: return specific ingredients.
        - For title_query or cooking_method_query: return key terms that help match relevant recipes. (delete "recipe" if there is)

        Your output should be a JSON object with the following structure:
        {{
            "intent": one of ["open_ended", "specific"],
            "type": ["title", "ingredients", "cooking_methods"],
            "title": {{
                "include": [...],
                "exclude": [...]
            }},
            "ingredients": {{
                "include": [...],
                "exclude": [...]
            }},
            "cooking_methods": {{
                "include": [...],
                "exclude": [...]
            }}
        }}  
        Only include keywords that are clearly stated or strongly implied in the query. 
        If a category does not apply, return an empty list for both `include` and `exclude`.

        Respond ONLY with a JSON object. Do not include any extra explanation or formatting.

        Query: {query}
        Response:
        """
    )
    def sort_and_join(keyword_list):
        return "; ".join(sorted(set(keyword_list)))
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key="AIzaSyDwaASu0nlEphtbTBpO8rW92bw43nHnwYw")

    classifier_chain = classifier_prompt | llm
    result = classifier_chain.invoke(input={"query": query})
    content = result.content.strip()
    # Remove ```json ... ``` or ``` ... ``` if present
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    
    print("Raw LLM output:")
    print(result.content)
    try:
        parsed = json.loads(content)
        for section in ["title", "ingredients", "cooking_methods"]:
            parsed[section]["include"] = sort_and_join(parsed[section]["include"])
        return parsed
    except json.JSONDecodeError:
        print("❌ Failed to parse JSON:")
        print(content)
        return {
            "type": [],
            "title": {
                "include": [],
                "exclude": []
            },
            "ingredients": {
                "include": [],
                "exclude": []
            },
            "cooking_methods": {
                "include": [],
                "exclude": []
            }
        }

if __name__ == "__main__":
    # Test 1: Recipe title query
    query_test1 = "Show me some recipes for making blueberry yogurt."
    print("\nTest 1: Recipe title query")
    res1 = query_classifier(query_test1)
    print(res1)

    # Test 2: Negation test
    query_test2 = "Can you find me a dessert recipe that does not have cherries and berries"
    print("\nTest 2: Negation test")
    res2 = query_classifier(query_test2)
    print(res2)

    # Test 3: General query
    query_test3 = "What to cook tonight?"
    print("\nTest 3: General query")
    res3 = query_classifier(query_test3)
    print(res3)
    
    
    # Test 4: Cooking method query
    query_test4 = "Show me recipes of beef that use boiling"
    print("\nTest 4: Cooking method query")
    res4 = query_classifier(query_test4)
    print(res4)
    

    # query_test4 = "Show me recipes of beef that use boiling"
    # res4 = query_classifier(query_test4)
    # print(res4)

    # # General test
    # query_test5 = "What to cook tonight?"
    # res5= query_classifier(query_test5)
    # print(res5)

    # Negation test
    # query_test6 = "I am allergic to banana. Can you find me a dessert recipe for me"
    # res6= query_classifier(query_test6)
    # print(res6)
