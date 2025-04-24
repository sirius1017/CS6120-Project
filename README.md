# Cooking Recipe Inspiration with RAG
This project implements a Retrieval-Augmented Generation (RAG) system that intelligently generates recipe suggestions based on natural language queries. Instead of relying solely on a large language model (LLM), our system first retrieves relevant recipes from a curated dataset and then uses the retrieved context to guide the final generation.

## Key Features
- Retrieval-Augmented Generation (RAG) with LangChain
- Hugging Face, Gemini API & Gemma3 model support
- Context-aware recipe generation
- JSON-based recipe filtering and querying

## Dataset 
We used Recipe1M+ as our dataset. The Recipe1M+ dataset is one of the largest publicly available collections of structured cooking recipes paired with images. Each recipe contains structered fileds such as title, ingredients, cooking instructions, nutririon values. 
The website of the Recipes1M+ can be found [**here**](https://im2recipe.csail.mit.edu/).


## Run Locally

1. **Clone the repo**
    ```bash
    git clone https://github.com/sirius1017/CS6120-Project.git
    cd CS6120-Project
    ```

2. **Set up `.env`**
    ```bash
    echo "GOOGLE_API_KEY=your_api_key_here" > .env
    echo "MODEL=gemma3:latest" >> .env
    ```

3. **Build using Docker**
    ```bash
    docker compose build
    docker compose up
    ```

4. **Access the app**

    Open your browser and go to: [http://localhost:8501](http://localhost:8501)

