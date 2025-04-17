from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from data_preprocessing import load_recipes_from_json
from typing import List, Dict

def retrieve_full_recipes(query: str,
                          mode: str,
                          json_path: str,
                          top_k: int = 5):
    full_recipes = load_recipes_from_json(json_path)
    id_to_recipe = {r["id"]: r for r in full_recipes}

    if mode == "title":
        persist_directory = "./chroma_title"
    elif mode == "ingredient":
        persist_directory = "./chroma_ingredients"
    else:
        print("input model wrong")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./hf_cache",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    results = vectorstore.similarity_search(query, k=top_k)

    matched = []
    for doc in results:
        recipe_id = doc.metadata.get("id")
        if recipe_id in id_to_recipe:
            matched.append(id_to_recipe[recipe_id])
        else:
            print(f"⚠️ Recipe id '{recipe_id}' not found in JSON.")

    return matched


def top_k_by_nutrient(recipes: List[Dict], 
                      nutrient: str, # "energy""fat""protein""salt""saturates""sugars"
                      k: int = 5, 
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
    return sorted_recipes[:k]


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
