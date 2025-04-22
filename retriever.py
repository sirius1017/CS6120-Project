from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from data_preprocessing import load_recipes_from_json
from typing import List, Dict
from collections import defaultdict
import numpy as np
from query_construction import query_classifier

# def retrieve_full_recipes(query: str,
#                           mode: str,
#                           json_path: str,
#                           top_k: int = 5):
#     full_recipes = load_recipes_from_json(json_path)
#     id_to_recipe = {r["id"]: r for r in full_recipes}

#     if mode == "title":
#         persist_directory = "./chroma_title"
#     elif mode == "ingredient":
#         persist_directory = "./chroma_ingredients"
#     else:
#         print("input model wrong")
#         return

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         cache_folder="./hf_cache",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True}
#     )

#     vectorstore = Chroma(
#         persist_directory=persist_directory,
#         embedding_function=embeddings
#     )

#     results = vectorstore.similarity_search(query, k=top_k)

#     matched = []
#     for doc in results:
#         recipe_id = doc.metadata.get("id")
#         if recipe_id in id_to_recipe:
#             matched.append(id_to_recipe[recipe_id])
#         else:
#             print(f"⚠️ Recipe id '{recipe_id}' not found in JSON.")

#     return matched


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def is_semantically_similar_to_exclude(recipe_text: str, exclude_vectors: List[List[float]], embeddings, threshold: float = 0.2) -> bool:
    try:
        recipe_vector = embeddings.embed_query(recipe_text)
    except Exception as e:
        print(f"⚠️ Embedding failed for recipe: {e}")
        return False

    for ex_vec in exclude_vectors:
        sim = cosine_similarity(recipe_vector, ex_vec)
        if sim >= threshold:
            return True
    return False


def retrieve_full_recipes(query: Dict,
                          json_path: str,
                          top_k: int = 5):
    full_recipes = load_recipes_from_json(json_path)
    id_to_recipe = {r["id"]: r for r in full_recipes}

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./hf_cache",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    modes = query["type"]
    exclude_keywords = (query["title"]["exclude"] + query["ingredients"]["exclude"] + query["instructions"]["exclude"])
    # Embed all exclusion terms once
    exclude_vectors = [embeddings.embed_query(term) for term in exclude_keywords]

    temp_results =[]

    for mode in modes:
        if mode == "title":
            persist_directory = "./chroma_title"
        elif mode == "ingredients":
            persist_directory = "./chroma_ingredients"
        elif mode == 'instructions':
            persist_directory = "./chroma_instructions"
        else:
            print(f"⚠️ Invalid mode: {mode}")
            continue

        # Load corresponding vectorstore
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        results = vectorstore.similarity_search(query[mode]["include"], k=top_k)
        temp_results.append(results)
    
    all_results = [doc for result_list in temp_results for doc in result_list]

    final_results = []
    for r in all_results:
        recipe_id = r.metadata.get("id")
        recipe = id_to_recipe[recipe_id]
        # recipe_text = f"{recipe.get('title', '')} {recipe.get('ingredients', '')}".lower()
        # if is_semantically_similar_to_exclude(recipe_text, exclude_vectors, embeddings):
        #     continue
        final_results.append(id_to_recipe[recipe_id])

    # Sort the recipes by nutrition
    if (query["nutritions"] is not None) and (query["descending"] is not None):
        final_results = top_k_by_nutrient(final_results, 
                               nutrient=query["nutritions"],  
                               descending = query["descending"]) 
    
    return final_results


def top_k_by_nutrient(recipes: List[Dict], 
                      nutrient: str, # "energy""fat""protein""salt""saturates""sugars"
                    #   k: int = 5, 
                      descending: bool = True # 默认降序，从大到小
                      ) -> List[Dict]:
    """按任意 nutrient 值排序（升序或降序），返回 top-k"""
    filtered = [
        r for r in recipes
        if "nutr_values_per100g" in r and nutrient in r["nutr_values_per100g"]
    ]
    sorted_recipes = sorted(
        filtered,
        key=lambda r: r["nutr_values_per100g"][nutrient],
        reverse=descending
    )
    return sorted_recipes

# def load_retriever(persist_directory: str = "./chroma_db"):
#     """Load ChromaDB-based retriever using HuggingFace embeddings."""
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         cache_folder="./hf_cache",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"batch_size": 128, "normalize_embeddings": True}
#     )

#     vectorstore = Chroma(
#         persist_directory=persist_directory,
#         embedding_function=embeddings
#     )

#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#     return retriever
if __name__ == "__main__":
    # query_test4 = "Can you find me a dessert recipe that does not have cherries and berries"
    # query= query_classifier(query_test4)
    # print(query)
    query={'type': ['title', 'ingredients'], 'title': {'include': 'yogurt', 'exclude': []}, 'ingredients': {'include': '', 'exclude': []}, 'instructions': {'include': '', 'exclude': []}, 'nutritions': None, 'descending': None}
    path = "./recipes.json"
    result = retrieve_full_recipes(query, path)
    
    for i, r in enumerate(result, start=1):
        title = r.get("title", "N/A")
        raw_ings = r.get("ingredients", [])
        ingredients = [i["text"].split(",")[0].lower() for i in raw_ings if "text" in i and i["text"]]
        url = r.get("url", "N/A")
        print(f"\n[{i}] {title}")
        print("Ingredients:", ", ".join(ingredients))
        print(f"url: {url}")

    print()