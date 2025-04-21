import json
from transformers import pipeline
from data_preprocessing import load_recipes_from_json

class CookingMethodExtractor:
    def __init__(self):
        # Using a zero-shot classification pipeline
        # This model can classify text into any set of labels we provide
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device="cpu"  # Use CPU for inference
        )
        
        # Define common cooking methods as classification labels
        self.cooking_methods = [
            "baking", "boiling", "broiling", "frying", "grilling",
            "roasting", "sautéing", "simmering", "steaming", "stir-frying",
            "toasting", "blending", "chopping", "mixing", "pureeing",
            "whisking", "marinating", "seasoning", "cooking", "heating",
            "microwaving", "pressure cooking", "slow cooking", "smoking",
            "deep frying", "braising", "poaching", "searing", "glazing"
        ]

    def extract_methods(self, instructions):
        """
        Extract cooking methods using zero-shot classification.
        
        Args:
            instructions (list): List of instruction objects with 'text' field
            
        Returns:
            list: A list of detected cooking methods with confidence scores
        """
        if not instructions:
            return []
        
        # Combine all instruction texts
        all_instructions = " ".join([instr.get("text", "") for instr in instructions])
        
        # Use zero-shot classification to identify cooking methods
        result = self.classifier(
            all_instructions,
            candidate_labels=self.cooking_methods,
            multi_label=True  # Allow multiple cooking methods to be detected
        )
        
        # Filter methods by confidence threshold
        threshold = 0.5  # Adjust this threshold based on testing
        detected_methods = [
            (label, score) 
            for label, score in zip(result["labels"], result["scores"]) 
            if score > threshold
        ]
        
        return detected_methods

def test_first_ten_recipes():
    """Test the extraction of cooking methods from the first ten recipes."""
    # Initialize the extractor
    extractor = CookingMethodExtractor()
    
    # Load recipes from JSON
    recipes = load_recipes_from_json("./recipes.json")
    
    # Test the first ten recipes
    for i, recipe in enumerate(recipes[:10]):
        print(f"\nRecipe {i+1}: {recipe.get('title', 'Untitled')}")
        
        # Get instructions
        instructions = recipe.get('instructions', [])
        if not instructions:
            print("  No instructions found.")
            continue
        
        # Extract cooking methods
        cooking_methods = extractor.extract_methods(instructions)
        
        # Print results
        if cooking_methods:
            print("  Cooking methods found:")
            for method, confidence in cooking_methods:
                print(f"    - {method} (confidence: {confidence:.2f})")
        else:
            print("  No cooking methods found.")
        
        # Print instructions for context
        print("\n  Instructions:")
        for instr in instructions[:2]:  # Show first two instructions
            print(f"    {instr.get('text', '')}")
        if len(instructions) > 2:
            print("    ...")

def compare_approaches():
    """Compare rule-based and model-based approaches on the same recipes."""
    from extract_cooking_methods import extract_cooking_methods  # Import the rule-based approach
    
    # Initialize the model-based extractor
    extractor = CookingMethodExtractor()
    
    # Load recipes
    recipes = load_recipes_from_json("./recipes.json")
    
    # Test the first five recipes with both approaches
    for i, recipe in enumerate(recipes[:5]):
        print(f"\n{'='*80}")
        print(f"Recipe {i+1}: {recipe.get('title', 'Untitled')}")
        
        instructions = recipe.get('instructions', [])
        if not instructions:
            print("  No instructions found.")
            continue
            
        # Test rule-based approach
        rule_based_methods = extract_cooking_methods(instructions)
        print("\nRule-based approach found:")
        print(f"  {', '.join(rule_based_methods) if rule_based_methods else 'None'}")
        
        # Test model-based approach
        model_based_methods = extractor.extract_methods(instructions)
        print("\nModel-based approach found:")
        for method, confidence in model_based_methods:
            print(f"  - {method} (confidence: {confidence:.2f})")
        
        # Print instructions for context
        print("\nInstructions:")
        for instr in instructions[:2]:
            print(f"  {instr.get('text', '')}")
        if len(instructions) > 2:
            print("  ...")

if __name__ == "__main__":
    print("Testing model-based extraction:")
    test_first_ten_recipes()
    
    print("\nComparing approaches:")
    compare_approaches() 