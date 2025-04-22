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
        - title: resembles a recipe name or dish (e.g., "how to make ramen", or includes health-based descriptors like "gluten-free", "keto", "lactose-free", "nut-free", "egg-free")
        - cooking_method: mentions a cooking technique (e.g., "boiling" or "baking")

        3. Extract key ingredients or concepts (as a list of keywords) from the query to assist in retrieving recipes. 
        - For ingredients_query: return specific ingredients.
        - For title_query or cooking_method_query: return key terms that help match relevant recipes. (delete "recipe" if there is)

        4. If the query requests a nutritional preference (e.g., "low fat", "high protein"), extract:
        - `"nutrition": <one of "energy", "fat", "protein", "salt", "saturates", "sugars">`
        - `"descending": true` if the user wants **high** value (e.g., "high protein")
        - `"descending": false` if the user wants **low** value (e.g., "low sugar")
        - If no nutrition preference is mentioned, return `"nutrition": null` and `"descending": null`

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
