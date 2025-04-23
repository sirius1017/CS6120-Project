from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os, json, re

load_dotenv()

def query_classifier(query):
    classifier_prompt = PromptTemplate.from_template(
        """You are a helpful AI cooking assistant. Your job is to analyze a user's recipe-related query and extract structured information for recipe retrieval.

        1. Classify the user query into the following types (can be multiple):
        - ingredients: mentions specific ingredients (e.g., "chicken and garlic")
        - title: resembles a recipe name or dish (e.g., "how to make ramen") OR refers to a broad category of food (e.g., "dessert", "salad", "pasta")
        - instructions: mentions cooking steps, methods, or requirements (e.g., "fry the chicken", "mix the milk and flour")

        3. Extract key requirements from corresponding types(as a list of keywords) from the query to assist in retrieving recipes. 
        - title: title: include the name of the dish or the general category (e.g., "dessert", "pasta") but do **not** include the word "recipe"
        - ingredients: return specific ingredients.
        - instructions: return key phrases that help match relevant recipes.
            - For instructions, preserve meaningful **action-ingredient** combinations, such as “cook milk, cream, and sugar” or “churn in ice cream maker”.
            - Exclude generic words like “recipe”, “dish”, or ambiguous phrasing.

        4. If the query requests a nutritional preference (e.g., "low fat", "high protein"), extract:
        - `"nutrition": <one of "energy", "fat", "protein", "salt", "saturates", "sugars">`
        - `"descending": true` if the user wants **high** value (e.g., "high protein")
        - `"descending": false` if the user wants **low** value (e.g., "low sugar")
        - If no nutrition preference is mentioned, return `"nutrition": None` and `"descending": None`

        5.  Ensure consistency:
        - If any category (title, ingredients, instructions) contains non-empty "include" or "exclude" lists, it must also appear in the top-level `"type"` list.

        Your output should be a JSON object with the following structure:
        {{
            "type": ["title", "ingredients", "instructions"],
            "title": {{
                "include": [...],
                "exclude": [...]
            }},
            "ingredients": {{
                "include": [...],
                "exclude": [...]
            }},
            "instructions": {{
                "include": [...],
                "exclude": [...]
            }},
            "nutritions": one of ["energy", "fat", "protein", "salt", "saturates", "sugars", None],
            "descending": true, false, or None
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
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key = os.getenv("GOOGLE_API_KEY"))
    classifier_chain = classifier_prompt | llm
    result = classifier_chain.invoke(input={"query": query})
    content = result.content.strip()
    # Remove ```json ... ``` or ``` ... ``` if present
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    
    try:
        parsed = json.loads(content)
        for section in ["ingredients"]:
            parsed[section]["include"] = sort_and_join(parsed[section]["include"])
        for section in ["title", "instructions"]:
            parsed[section]["include"] = ",".join(parsed[section]["include"])
            
        print(parsed)
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
            "instructions": {
                "include": [],
                "exclude": []
            },
            "nutritions": None,
            "descending": None
        }

if __name__ == "__main__":
    # query_test1 = "Show me some recipes for making blueberry yogurt."
    # res1= query_classifier(query_test1)
    # print(res1)

    # query_test2 = "Show me recipes of beef that use boiling"
    # res2 = query_classifier(query_test2)
    # print(res2)

    # query_test3 = "Give me recipes using beef and potatoes"
    # res3= query_classifier(query_test3)
    # print(res3)

    # # Negation test
    # query_test4 = "Can you find me a dessert recipe that does not have cherries and berries"
    # res4= query_classifier(query_test4)
    # print(res4)

    # # General test
    # query_test5 = "What to cook tonight?"
    # res5= query_classifier(query_test5)
    # print(res5)

    # Negation test
    # query_test6 = "I am allergic to banana. Can you find me a dessert recipe for me"
    # res6= query_classifier(query_test6)
    # print(res6)

    # Long query test
    query_test7 = "I've seen a recipe calls for cooking the milk,\
          cream, and sugar until the sugar has dissolved. Then, we would mix with a cup, \
            while adding vanilla extract. We need an ice cream maker for churning according to the manufacturer's directions, \
                but I don't know if I have it. We would need to serve immediately or ripen in the freezer. \
                    Do you know what this recipe is for?"
    res7 = query_classifier(query_test7)
    print(res7)
