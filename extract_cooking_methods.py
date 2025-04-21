import json
import re
from data_preprocessing import load_recipes_from_json

# Common cooking methods to look for in instructions
COMMON_COOKING_METHODS = [
    "bake", "baking", "boil", "boiling", "broil", "broiling", "fry", "frying", 
    "grill", "grilling", "roast", "roasting", "sauté", "sautéing", "simmer", "simmering",
    "steam", "steaming", "stir-fry", "stir-frying", "toast", "toasting", "blend", "blending",
    "chop", "chopping", "dice", "dicing", "grate", "grating", "mix", "mixing", "puree", "pureeing",
    "whisk", "whisking", "marinate", "marinating", "season", "seasoning", "cook", "cooking",
    "heat", "heating", "preheat", "preheating", "microwave", "microwaving", "pressure cook", "pressure cooking",
    "slow cook", "slow cooking", "smoke", "smoking", "deep fry", "deep frying", "pan fry", "pan frying",
    "braise", "braising", "poach", "poaching", "sear", "searing", "glaze", "glazing", "ferment", "fermenting"
]

def extract_cooking_methods(instructions):
    """
    Extract cooking methods from recipe instructions.
    
    Args:
        instructions (list): List of instruction objects with 'text' field
        
    Returns:
        list: A list of cooking methods found in the instructions
    """
    if not instructions:
        return []
    
    # Combine all instruction texts
    all_instructions = " ".join([instr.get("text", "") for instr in instructions])
    
    # Convert to lowercase for case-insensitive matching
    instructions_lower = all_instructions.lower()
    
    # Find all cooking methods in the instructions
    found_methods = []
    for method in COMMON_COOKING_METHODS:
        # Look for the method as a whole word
        pattern = r'\b' + re.escape(method) + r'\b'
        if re.search(pattern, instructions_lower):
            found_methods.append(method)
    
    return found_methods

def test_first_ten_recipes():
    """Test the extraction of cooking methods from the first ten recipes."""
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
        cooking_methods = extract_cooking_methods(instructions)
        
        # Print results
        print(f"  Cooking methods found: {', '.join(cooking_methods) if cooking_methods else 'None'}")
        
        # Print a snippet of instructions for context
        first_instruction = instructions[0].get("text", "") if instructions else ""
        print(f"  First instruction: {first_instruction[:100]}...")

if __name__ == "__main__":
    test_first_ten_recipes() 