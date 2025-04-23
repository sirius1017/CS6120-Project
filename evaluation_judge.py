import datetime
from generator_response import generate_response
from ollama import chat
import os
import csv
import re


def generate_no_rag_response(query):
    """Use only the LLM without RAG context."""
    direct_prompt = f"""
    You are a helpful recipe assistant. A user asked:

    "{query}"

    Based on your knowledge, generate 1 recipe that might answer the user's question.

    Present them in the following format:
    
    1. Recipe Title:
    [Title]

    2. Ingredients:
    [List of ingredients with precise measurements]

    3. Cooking Instructions:
    [Step-by-step instructions]

    4. Nutritional Information:
    [Estimates per 100g]

    5. Cooking Tips:
    [Tips for best results]
    """

    response = chat(
        model="gemma3:latest",
        messages=[{"role": "user", "content": direct_prompt}]
    )
    return response['message']['content']


def ask_llm_judge(query, rag_response, llm_response):
    judge_prompt = f"""
You are an expert food and recipe evaluator. Given a user query and two responses (one generated using retrieved context and the other using only LLM knowledge), evaluate both based on the following criteria:

1. **Ingredient Accuracy**: Are the ingredients realistic and appropriate for the dish?
2. **Proportion Reasonability**: Are the ingredient quantities balanced and suitable?
3. **Query Relevance**: Does the recipe answer the user’s request (even if it's vague or long)?
4. **Preparation Logic**: Are the cooking steps logical, safe, and practical?
5. **Constraint Handling**: For queries that mention negations or exclusions (e.g., "without nuts"), are those constraints followed correctly?

Please score each response (RAG and LLM-only) on a scale from 1 to 5 per category.
Then choose a winner and provide a short justification.

### Query:
{query}

### RAG Response:
{rag_response}

### LLM-Only Response:
{llm_response}

Respond in this format:

```
RAG Scores: [ingredient_accuracy, proportions, relevance, logic, constraints]
LLM Scores: [ingredient_accuracy, proportions, relevance, logic, constraints]
Winner: ["RAG" or "LLM"]
Justification: <Your explanation here>
```    
    """

    result = chat(
        model="gemma3:latest",
        messages=[{"role": "user", "content": judge_prompt}]
    )
    return result['message']['content']

def parse_judge_output(text):
    try:
        rag_scores = re.search(r"RAG Scores:\s*\[(.*?)\]", text).group(1).split(",")
        llm_scores = re.search(r"LLM Scores:\s*\[(.*?)\]", text).group(1).split(",")
        winner = re.search(r"Winner:\s*\[\"?(RAG|LLM)\"?\]", text).group(1)
        justification = re.search(r"Justification:\s*(.*)", text, re.DOTALL).group(1).strip()

        rag_scores = [int(s.strip()) for s in rag_scores]
        llm_scores = [int(s.strip()) for s in llm_scores]

        return rag_scores, llm_scores, winner, justification
    except Exception as e:
        print("❌ Failed to parse judge response:", e)
        print("Raw Response:\n", text)
        return [0]*5, [0]*5, "Unknown", "Parsing failed"


def run_evaluation():
    test_queries = [
        "Give me a dessert without nuts and milk",
        "How to make ice cream without an ice cream maker?",
        "I want a vegetarian pasta recipe high in protein",
        "Find me a recipe that includes yogurt but no fruit",
        "What to cook of dinner tonight",
        "I've seen a recipe calls for cooking the milk,\
          cream, and sugar until the sugar has dissolved. Then, we would mix with a cup, \
            while adding vanilla extract. We need an ice cream maker for churning according to the manufacturer's directions, \
                but I don't know if I have it. We would need to serve immediately or ripen in the freezer. \
                    Do you know what this recipe is for?",
        "I want something sweet",
        "What can I cook with no dairry n no meat?",
        "Give me a healthy dinner",
        "I feel like eating something cozy",
        "Can you give me a cherry cake without any cherries?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n=== Test Case {i} ===")
        print(f"Query: {query}\n")

        rag_resp = generate_response(query)
        llm_resp = generate_no_rag_response(query)

        print("\n--- RAG Response ---\n")
        print(rag_resp)

        print("\n--- LLM-Only Response ---\n")
        print(llm_resp)

        print("\n--- LLM Judge Evaluation ---")
        judge_result = ask_llm_judge(query, rag_resp, llm_resp)
        print(judge_result)

    

if __name__ == "__main__":
    run_evaluation()
