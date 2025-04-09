# CS6120-Project
This is final project for CS 6120 NLP


# install

conda create --name rag python=3.9
conda activate rag
pip install langchain chromadb tiktoken python-dotenv
pip install -U langchain-community
pip install sentence-transformers
pip install -U langchain-huggingface
pip install transformers accelerate

# how to run

run python app.py to use the RAG
run python test_model.py to use the LLM without RAG
