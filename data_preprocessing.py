import json
from typing import List, Dict
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings # 使用huggingface而不用openai，这样可以直接把模型下载到本地使用，不用call api
import os
import shutil
from typing import Dict, List
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from tqdm import tqdm
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv

#调优：encoder现在用的是HuggingFaceEmbeddings，可能可以换一下试试？
#加速：batch embedding, batch write to chroma db
#threshold：test用，最后需要去掉
#计时

def summarize_instructions(recipes, path):
    summarization_prompt = PromptTemplate.from_template("""
        You are a recipe assistant. Your job is to summarize the following recipe instructions.

        Focus on:
        - The main **cooking methods**
        - Important **kitchen tools**
        - Key preparation or serving steps

        Instructions:
        {instructions}

        Return a concise summary that highlights the above elements.
        """)
    
    # Set up HuggingFace summarizer model 
    # device = 0, run gpu
    flan_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=0, max_length=128)

    # Wrap it for LangChain compatibility
    flan_llm = HuggingFacePipeline(pipeline=flan_pipeline)
    chain = summarization_prompt | flan_llm
    
    summaries = []
    for recipe in tqdm(recipes):
        raw_steps = " ".join([step["text"] for step in recipe.get("instructions", [])])
        if not raw_steps.strip():
            continue

        try:
            summary = chain.invoke(input={"instructions": raw_steps})
            summaries.append({
                "id": recipe.get("id", ""),
                "summary": summary
            })
        except Exception as e:
            print(f"❌ Failed for {recipe.get('id')}: {e}")

    # Save to JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(summaries)} summaries to {path}")

def load_or_create_vectorstore(docs, embeddings, persist_dir):
    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        print(f"✅ Loading existing vectorstore from: {persist_dir}")
        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
    else:
        print(f"📚 Creating and saving new vectorstore to: {persist_dir}")
        return Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )

def prepare_documents(recipes: List[dict]) -> List:
    title_docs = []
    ingredients_docs = []
    instruction_docs = []

    for recipe in recipes:
        title = recipe.get("title", "Untitled")

        raw_ings = recipe.get("ingredients", [])

        # # 只提取各ingredient的第一个单词
        # # 存在的问题 eg. "candies, semisweet chocolate"
        first_words = [i["text"].split(",")[0].lower() for i in raw_ings if "text" in i and i["text"]]
        ingredients = ";".join(sorted(first_words))

        recipe_id = recipe.get("id", "")
        metadata = { 
            "id": recipe_id
        }

        title_docs.append(Document(page_content=title, metadata=metadata)) 
        ingredients_docs.append(Document(page_content=ingredients, metadata=metadata))

        # ----- Step-Level Instructions -----
        instruction_texts = [step['text'] for step in recipe.get('instructions', [])]
        for idx, text in enumerate(instruction_texts):
            if text.strip():
                instruction_docs.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "id": recipe_id,
                        "step": idx
                    }
                ))

        # instruction_texts = [step['text'] for step in recipe.get('instructions', [])]
        # full_instructions = " ".join(instruction_texts)
        # if full_instructions.strip():  # skip empty
        #     instruction_docs.append(Document(page_content=full_instructions, metadata=metadata))

    return title_docs, ingredients_docs, instruction_docs


def load_recipes_from_json(file_path: str) -> List[Dict]:
    """Load and parse recipes from a local JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        recipes = json.load(f)
    return recipes


def ingest_to_chroma(json_path="./recipes.json", 
                    persist_dir_title="./chroma_title", 
                    persist_dir_ingredients="./chroma_ingredients",
                    persist_dir_instructions="./chroma_instructions",
                    threshold=-1):
    """Main pipeline: load -> format -> embed -> save to ChromaDB."""

    print("🔍 Loading recipes...")

    # # 先删除旧向量库
    # for path in [persist_dir_title, persist_dir_ingredients]:
    #     if os.path.exists(path):
    #         print(f"⚠️ Found existing directory {path}. Removing it before regeneration...")
    #         shutil.rmtree(path)

    recipes = load_recipes_from_json(json_path)
    if threshold > 0:
        recipes = recipes[:threshold]

    title_docs, ingredients_docs, instructions_docs = prepare_documents(recipes)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./hf_cache",
        model_kwargs={"device": "cuda"}, ###################### GPU
        encode_kwargs={"batch_size": 128, "normalize_embeddings": True}
    )

    print("📚 Embedding titles...")
    title_store = load_or_create_vectorstore(title_docs, embeddings, persist_dir_title)

    print("🥬 Embedding ingredients...")
    ingredients_store = load_or_create_vectorstore(ingredients_docs, embeddings, persist_dir_ingredients)

    print("🥬 Embedding instructions...")
    instructions_store = load_or_create_vectorstore(instructions_docs, embeddings, persist_dir_instructions)


if __name__ == "__main__":
    with open("./recipes.json", "r", encoding="utf-8") as f:
        recipes = json.load(f)
    print(f"一共 {len(recipes)} 条菜谱")

    # # ingest_to_chroma(threshold=100) #试运行转换为embeddings
    ingest_to_chroma() # recipes--> embeddings
    # summarize_instructions(recipes, "./summarized_instructions.json")

