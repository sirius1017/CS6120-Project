from generator_response import generate_response
from SUPPORTED_MODELS import SUPPORTED_MODELS 
from query_construction import query_classifier
from retriever import retrieve_full_recipes
import streamlit as st
import re, os, json


def retrive_recipes(input:str):
    print(f"Processing query: {input}")
    query = query_classifier(input)
    print("Query after classification:")
    print(query)
    
    # # Fix data structure to match what retrieve_full_recipes2 expects
    # for category in ["title", "ingredients", "cooking_methods"]:
    #     if isinstance(query[category]["include"], list) and len(query[category]["include"]) > 0:
    #         # Join list items into a single string
    #         query[category]["include"] = ';'.join(query[category]["include"])
    #     elif isinstance(query[category]["include"], list) and len(query[category]["include"]) == 0:
    #         # Empty list becomes empty string
    #         query[category]["include"] = ""
    
    # print("Query after fixing data types:")
    # print(query)
    
    ids = retrieve_full_recipes(query, "./recipes.json")
    print(f"Number of recipes found: {len(ids)}")
    return ids, query  # Return the query too
    
def retrive_recipes_title(query:str):
    query["type"] = ["title"]
    ids = retrieve_full_recipes(query,"./recipes.json")
    return ids

def retrive_recipes_ingredients(query:str):
    query["type"] = ["ingredients"]
    ids = retrieve_full_recipes(query,"./recipes.json")
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


def format_recipe(recipe):
    title = recipe.get("title", "N/A")
    raw_ings = recipe.get("ingredients", [])
    ingredients = [i["text"].split(",")[0].lower() for i in raw_ings if "text" in i and i["text"]]

    nutrition = recipe.get("nutr_values_per100g", {})
    def fmt(x): return f"{float(x):.2f}" if isinstance(x, (int, float)) else "N/A"

    energy = fmt(nutrition.get("energy"))
    fat = fmt(nutrition.get("fat"))
    protein = fmt(nutrition.get("protein"))
    salt = fmt(nutrition.get("salt"))
    saturates = fmt(nutrition.get("saturates"))
    sugars = fmt(nutrition.get("sugars"))

    return f"""### 🍽️ {title}
**Ingredients:** {", ".join(ingredients)}

**Nutrition per 100g:**
- Energy: {energy}
- Fat: {fat}
- Protein: {protein}
- Salt: {salt}
- Saturates: {saturates}
- Sugars: {sugars}
"""


def clean_generated_recipe(recipe_text):
    """Clean up the formatting of a generated recipe."""
    # Remove image references
    recipe_text = re.sub(r'<img[^>]*>', '', recipe_text)
    recipe_text = re.sub(r'!\[\]?\([^)]*\)', '', recipe_text)
    recipe_text = re.sub(r'!\[\[[^\]]*\]\]', '', recipe_text)
    
    # Remove source citations
    recipe_text = re.sub(r'source \^\([^)]*\)', '', recipe_text)
    recipe_text = re.sub(r'\[\s*source\s*\]', '', recipe_text)
    recipe_text = re.sub(r'\^Source:[^\n]*', '', recipe_text)
    
    # Remove other markdown artifacts
    recipe_text = re.sub(r'--\[\[[^\]]*\]\]', '', recipe_text)
    
    # Fix formatting issues
    recipe_text = re.sub(r'\*{3,}', '**', recipe_text)  # Fix excessive asterisks
    recipe_text = re.sub(r'\.o\.', '', recipe_text)  # Remove strange notation
    
    # Remove strange annotations
    recipe_text = re.sub(r'\[\([^)]*\)\]', '', recipe_text)
    recipe_text = re.sub(r'\[\[[^\]]*\]\]', '', recipe_text)
    
    # Remove programming code patterns
    recipe_text = re.sub(r'#include.*', '', recipe_text)
    recipe_text = re.sub(r'int main\(\).*', '', recipe_text)
    recipe_text = re.sub(r'printf\(.*\);', '', recipe_text)
    recipe_text = re.sub(r'scanf\(.*\);', '', recipe_text)
    recipe_text = re.sub(r'return [0-9];', '', recipe_text)
    recipe_text = re.sub(r'char [a-zA-Z_]+ *\[[0-9]+\] *;', '', recipe_text)
    recipe_text = re.sub(r'/\*.*?\*/', '', recipe_text, flags=re.DOTALL)  # Remove C-style comments
    recipe_text = re.sub(r'//.*', '', recipe_text)  # Remove C++ style comments
    
    # Clean up multiple spaces and line breaks
    recipe_text = re.sub(r'\s{2,}', ' ', recipe_text)
    recipe_text = re.sub(r'\n{3,}', '\n\n', recipe_text)
    
    return recipe_text


def format_generated_recipe(recipe_text):
    """Format the generated recipe into structured sections with better parsing."""
    # Check for code patterns and filter them out
    if re.search(r'#include|int main|\bprintf\b|\bscanf\b|\breturn 0;|\bchar\b', recipe_text):
        return "Invalid recipe format detected. Please try another query."
    
    # Clean the recipe text
    recipe_text = re.sub(r'[^\w\s.,;:()/\-\'"%]+', ' ', recipe_text)  # Remove all special characters except common ones
    recipe_text = re.sub(r'\s{2,}', ' ', recipe_text)  # Remove extra spaces
    
    # Extract title
    title_match = re.search(r'(?:Title:|Recipe Title:|New Recipe:)\s*([^\n]+)', recipe_text)
    title = title_match.group(1).strip() if title_match else "Personalized Recipe"
    
    # Extract ingredients - look for cleaner separation
    ingredients_match = re.search(r'Ingredients:(.*?)(?:Instructions:|Directions:|Steps:|$)', recipe_text, re.DOTALL | re.IGNORECASE)
    ingredients_text = ingredients_match.group(1).strip() if ingredients_match else ""
    
    # Clean and extract actual ingredients (before instructions begin)
    ingredients_list = []
    if ingredients_text:
        # Try to find where instructions begin (look for keywords)
        instructions_keywords = ['make sure', 'prepare', 'preheat', 'heat', 'cook', 'bake', 'mix', 'combine', 'stir']
        cutoff_index = len(ingredients_text)
        
        for keyword in instructions_keywords:
            keyword_match = re.search(rf'\b{keyword}\b', ingredients_text.lower())
            if keyword_match and keyword_match.start() < cutoff_index:
                cutoff_index = keyword_match.start()
        
        # Get just the ingredients part
        clean_ingredients = ingredients_text[:cutoff_index].strip()
        
        # Split into items (by commas or newlines)
        for item in re.split(r'[,\n]', clean_ingredients):
            item = item.strip()
            if item and len(item) > 2 and not item.startswith('Enter code'):
                ingredients_list.append(item)
    
    # Extract instructions
    instructions_match = re.search(r'(?:Instructions:|Directions:|Steps:)(.*)', recipe_text, re.DOTALL | re.IGNORECASE)
    instructions_text = instructions_match.group(1).strip() if instructions_match else ""
    
    # If no formal instructions section, try to extract from the text
    if not instructions_text and ingredients_text:
        instructions_start = 0
        for keyword in ['make sure', 'prepare', 'preheat', 'heat', 'cook', 'bake', 'mix', 'combine', 'stir']:
            keyword_match = re.search(rf'\b{keyword}\b', ingredients_text.lower())
            if keyword_match and keyword_match.start() > 0:
                instructions_start = keyword_match.start()
                instructions_text = ingredients_text[instructions_start:].strip()
                break
    
    # Format everything nicely with smaller text
    formatted_recipe = f"""### {title}

**Ingredients:**  
"""
    # Display ingredients
    if ingredients_list:
        formatted_recipe += ", ".join(ingredients_list[:10])
        if len(ingredients_list) > 10:
            formatted_recipe += f", plus {len(ingredients_list) - 10} more ingredients"
    else:
        formatted_recipe += "No specific ingredients found in recipe."
    
    # Add instructions - parse into bullet points if possible
    formatted_recipe += "\n\n**Instructions:**  \n"
    if instructions_text:
        # Try to break into steps
        steps = []
        
        # Look for numbered steps
        numbered_steps = re.findall(r'\b(\d+)[.)] *(.*?)(?=\b\d+[.)]|$)', instructions_text, re.DOTALL)
        if numbered_steps:
            for num, step in numbered_steps:
                steps.append(step.strip())
        else:
            # Try to split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', instructions_text)
            for sentence in sentences:
                if len(sentence) > 10:  # Avoid tiny fragments
                    steps.append(sentence.strip())
        
        # Format as bullet points for readability
        if steps:
            for step in steps[:5]:  # Limit to 5 steps to keep it compact
                formatted_recipe += f"• {step}  \n"
            if len(steps) > 5:
                formatted_recipe += f"• Plus {len(steps) - 5} more steps..."
        else:
            # Just use the raw text if we couldn't parse steps
            formatted_recipe += instructions_text[:300] + "..."
    else:
        formatted_recipe += "Detailed cooking instructions not available."
    
    return formatted_recipe


if __name__ == "__main__":
    # Debug: Check if recipes.json exists

    recipes_path = "./recipes.json"
    if os.path.exists(recipes_path):
        try:
            with open(recipes_path, 'r') as f:
                recipes_data = json.load(f)
                print(f"Recipe file exists and contains {len(recipes_data)} recipes.")
        except json.JSONDecodeError:
            print("Recipe file exists but contains invalid JSON.")
        except Exception as e:
            print(f"Error reading recipe file: {str(e)}")
    else:
        print(f"Recipe file does not exist at path: {recipes_path}")
    
    # Load the generator model on startup (can be CPU if no GPU)
    try:
        device = "cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu"
        # generator = load_generator(model_key="deepseek", device=device)
        generator_loaded = True
        print("Recipe generator loaded successfully!")
    except Exception as e:
        generator_loaded = False
        print(f"Could not load recipe generator: {str(e)}")

    # ------------------
    # 🔧 初始化
    # ------------------

    st.set_page_config(page_title="Smart Recipe Finder", page_icon="🍳")
    
    # Custom CSS for better text sizing
    st.markdown("""
    <style>
    .recipe-header h4 {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .recipe-content p, .recipe-content li {
        font-size: 0.9rem;
        line-height: 1.3;
    }
    .recipe-content ul {
        margin-top: 0.3rem;
        padding-left: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🍳 Smart Recipe Finder")
    st.markdown("Ask me anything about recipes — ingredients, cooking styles, dietary preferences, etc.")

    # ------------------
    # 📥 用户输入
    # ------------------
    user_input = st.text_input("🤔 What's your craving today?", placeholder="e.g. something high in protein without sugar")

    if user_input:
        # First section: Generated recipe
        if generator_loaded:
            with st.spinner("🧪 Creating a personalized recipe..."):
                try:
                    # Generate the recipe
                    generated_recipe = generate_response(user_input, "./recipes.json")
                    
                    # Clean up the generated recipe
                    clean_recipe = clean_generated_recipe(generated_recipe)
                    
                    # Detect if this looks like code instead of a recipe
                    code_pattern = re.compile(r'#include|int main|\bprintf\b|\bscanf\b|\breturn 0;|\bchar\b')
                    if code_pattern.search(clean_recipe):
                        st.error("The recipe generator produced code instead of a recipe. Please try a different query.")
                    else:
                        # Format the recipe into a more structured display
                        formatted_recipe = format_generated_recipe(clean_recipe)
                        
                        # Only display the recipe if it's valid
                        if formatted_recipe != "Invalid recipe format detected. Please try another query.":
                            st.markdown("## 🧪 Personalized Recipe Suggestion")
                            
                            # Display recipe with custom HTML/CSS for smaller text
                            st.markdown(f'<div class="recipe-header recipe-content">{formatted_recipe}</div>', unsafe_allow_html=True)
                            
                            st.divider()  # Add a divider between generated and retrieved recipes
                        else:
                            st.warning("Could not generate a valid recipe. Showing only existing recipes.")
                except Exception as e:
                    st.error(f"Could not generate recipe: {str(e)}")
                    import traceback
                    print(f"Recipe generation error: {traceback.format_exc()}")
        else:
            st.warning("Recipe generator is not available, showing only existing recipes.")
        
        

        # Second section: Retrieved recipes
        with st.spinner("🔎 Searching recipes..."):
            # results = retrive_recipes(user_input)
            results, query = retrive_recipes(user_input)

        if not results:
            st.warning("No recipes found. Try a different query.")
        else:
            st.success(f"Found {len(results)} matching recipes.")
            for r in results:
                st.markdown(format_recipe(r))

            # ➕ 生成摘要回答按钮
            if st.button("💬 Summarize Suggestions"):
                with st.spinner("Generating helpful answer..."):
                    try:
                        # answer = generate_answer(user_input, results, generator)
                        answer = "answer"
                        st.markdown("### 🤖 Assistant Suggests")
                        st.success(answer)
                    except Exception as e:
                        st.error(f"Could not generate summary: {str(e)}")



# if __name__ == "__main__":
    
#     retriever = load_retriever()
#     model_key = "deepseek"
#     # model_key = "tinyllama"

#     generator = load_generator(model_key)
#     print(f'Using model {SUPPORTED_MODELS[model_key]["hf_id"]}')

#     while True:
#         print("🧑‍🍳 RAG chef is ready. Ask your question! (Type 'exit' to quit)")
#         query = input("🤔 You: ")
#         if query.lower() in {"exit"}:
#             break
#         print("🤖 Generating...\n")
        
#         docs = retriever.invoke(query)

#         answer = generate_answer(query, docs, generator)
#         print("🍽️ Answer:", answer)
    

#     print("🍳 Welcome to the Smart Recipe Finder!")
    
#     while True:
#         print("\nWhat would you like to do?")
#         print("1. Search recipes (general)")
#         print("2. Search by title only")
#         print("3. Search by ingredients only")
#         print("0. Exit")

#         choice = input("Enter your choice (1-5): ").strip()

#         if choice == "1":
#             user_query = input("Enter your recipe query: ")
#             results = retrive_recipes(user_query)
#             print_recipes(results)

#         elif choice == "2":
#             user_query = input("Enter title-related query: ")
#             structured = query_classifier(user_query)
#             results = retrive_recipes_title(structured)
#             print_recipes(results)

#         elif choice == "3":
#             user_query = input("Enter ingredients-related query: ")
#             structured = query_classifier(user_query)
#             results = retrive_recipes_ingredients(structured)
#             print_recipes(results)

#         elif choice == "0":
#             print("👋 Exiting. Enjoy your cooking!")
#             break
#         else:
#             print("❌ Invalid option. Please choose 1-5.")
