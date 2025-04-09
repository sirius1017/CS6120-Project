import json
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings # 使用huggingface而不用openai，这样可以直接把模型下载到本地使用，不用call api

#怎么把data转换成embeddings
#调优：每个recipe作为一个chunk而不是以长度分段
#调优：encoder现在用的是HuggingFaceEmbeddings，可能可以换一下试试
#加速：batch embedding, batch write to chroma db
#threshold：test用，最后需要去掉
#避免需要手动清空chroma_db的情况

def load_recipes_from_json(file_path: str) -> List[Dict]:
    """Load and parse recipes from a local JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        recipes = json.load(f)
    return recipes

def tag_metadata(recipe: dict) -> dict:
    """Convert metadata fields to simple types (str) for Chroma."""
    ingredients = "; ".join([i["text"] for i in recipe.get("ingredients", [])])
    instructions = " ".join([s["text"] for s in recipe.get("instructions", [])])

    return {
        "title": recipe.get("title", "Untitled"),
        "ingredients": ingredients,
        "instructions": instructions
    }


def format_recipe(recipe: dict) -> Document:
    """Convert raw recipe data into a LangChain Document (no URL)."""
    title = recipe.get("title", "Untitled")

    ingredients = "\n".join(
        [f"- {q['text']} {u['text']} {i['text']}" for q, u, i in zip(
            recipe.get("quantity", []),
            recipe.get("unit", []),
            recipe.get("ingredients", [])
        )]
    )

    instructions = "\n".join([step["text"] for step in recipe.get("instructions", [])])

    content = f"""Title: {title}

Ingredients:
{ingredients}

Instructions:
{instructions}
"""
    metadata = tag_metadata(recipe)
    return Document(page_content=content.strip(), metadata=metadata)


def ingest_to_chroma(json_path="./recipes.json", persist_directory="./chroma_db", threshold=-1):
    """Main pipeline: load -> format -> embed -> save to ChromaDB."""

    print("🔍 Loading recipes...")

    recipes = load_recipes_from_json(json_path)
    if threshold > 0:
        recipes = recipes[:threshold]

    documents = [format_recipe(r) for r in recipes]
    print(f"📄 Total documents (recipes): {len(documents)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./hf_cache",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 128, "normalize_embeddings": True}
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print("✅ Embedding complete and saved to ChromaDB!")


if __name__ == "__main__":
    ingest_to_chroma(threshold=100)
