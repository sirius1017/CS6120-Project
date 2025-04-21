from generator import load_generator, generate_answer

def test_generator():
    # Load the generator
    generator = load_generator(model_key="deepseek", device="cuda")
    
    # Test query
    test_query ="Can you find me a dessert recipe that does not have cherries and berries?"
    
    try:
        # Generate answer using the retriever
        response = generate_answer(test_query, generator)
        print("\nTest Query:", test_query)
        print("\nGenerated Response:", response)
        return True
    except Exception as e:
        print("Error during test:", str(e))
        return False

if __name__ == "__main__":
    test_generator() 