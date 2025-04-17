import json
from typing import List, Dict
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings # 使用huggingface而不用openai，这样可以直接把模型下载到本地使用，不用call api
import os
import shutil
from typing import Dict, List

#调优：encoder现在用的是HuggingFaceEmbeddings，可能可以换一下试试
#加速：batch embedding, batch write to chroma db
#threshold：test用，最后需要去掉
#计时

def prepare_documents(recipes: List[dict]) -> (List[Document], List[Document]):
    title_docs = []
    ingredients_docs = []

    for recipe in recipes:
        title = recipe.get("title", "Untitled")

        raw_ings = recipe.get("ingredients", [])
        # !!!! 目前只提取各ingredient的第一个单词，但是遇到某些特殊数据出现问题
        # eg. "candies, semisweet chocolate"
        first_words = [i["text"].split(",")[0].lower() for i in raw_ings if "text" in i and i["text"]]
        ingredients = ";".join(sorted(first_words))

        recipe_id = recipe.get("id", "")
        metadata = { 
            "id": recipe_id
        }

        title_docs.append(Document(page_content=title, metadata=metadata)) 
        ingredients_docs.append(Document(page_content=ingredients, metadata=metadata))

    return title_docs, ingredients_docs


def load_recipes_from_json(file_path: str) -> List[Dict]:
    """Load and parse recipes from a local JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        recipes = json.load(f)
    return recipes


def ingest_to_chroma(json_path="./recipes.json", 
                    #  persist_directory="./chroma_db", 
                    persist_dir_title="./chroma_title", 
                    persist_dir_ingredients="./chroma_ingredients",
                    threshold=-1):
    """Main pipeline: load -> format -> embed -> save to ChromaDB."""

    print("🔍 Loading recipes...")

    # 先删除旧向量库
    for path in [persist_dir_title, persist_dir_ingredients]:
        if os.path.exists(path):
            print(f"⚠️ Found existing directory {path}. Removing it before regeneration...")
            shutil.rmtree(path)

    recipes = load_recipes_from_json(json_path)
    if threshold > 0:
        recipes = recipes[:threshold]

    title_docs, ingredients_docs = prepare_documents(recipes)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./hf_cache",
        model_kwargs={"device": "cpu"}, ###################### GPU
        encode_kwargs={"batch_size": 128, "normalize_embeddings": True}
    )

    print("📚 Embedding titles...")
    title_store = Chroma.from_documents(
        documents=title_docs,
        embedding=embeddings,
        persist_directory=persist_dir_title
    )

    print("🥬 Embedding ingredients...")
    ingredients_store = Chroma.from_documents(
        documents=ingredients_docs,
        embedding=embeddings,
        persist_directory=persist_dir_ingredients
    )

    print(f"✅ Data saved to: {persist_dir_title} and {persist_dir_ingredients}")



if __name__ == "__main__":
    with open("./recipes.json", "r", encoding="utf-8") as f:
        recipes = json.load(f)
    print(f"一共 {len(recipes)} 条菜谱")

    # ingest_to_chroma(threshold=100) #试运行转换为embeddings
    ingest_to_chroma() # recipes--> embeddings




