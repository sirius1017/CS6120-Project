from generator_response import generate_response
import streamlit as st
import re
import os, json
import random

if __name__ == "__main__":
    # Debug: Check if recipes.json exists
    recipes_path = "./recipes.json"
    if os.path.exists(recipes_path):
        try:
            with open(recipes_path, 'r') as f:
                recipes_data = json.load(f)
                print(f"Recipe file exists and contains {len(recipes_data)} recipes.")
        except json.JSONDecodeError:
            print("Recipe file exists but contains invalid JSON.")
        except Exception as e:
            print(f"Error reading recipe file: {str(e)}")
    else:
        print(f"Recipe file does not exist at path: {recipes_path}")
    
    # Load the generator model on startup (can be CPU if no GPU)
    try:
        device = "cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu"
        # generator = load_generator(model_key="deepseek", device=device)
        generator_loaded = True
        print("Recipe generator loaded successfully!")
    except Exception as e:
        generator_loaded = False
        print(f"Could not load recipe generator: {str(e)}")

    # ------------------
    # 🔧 Initialization
    # ------------------

    st.set_page_config(page_title="Smart Recipe Generator", page_icon="🍳")
    
    # Custom CSS for styling like the screenshot
    st.markdown("""
    <style>
    /* Overall page styling */
    .main {
        padding: 1.5rem;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Header styling */
    .header-container {
        background-color: #e8f5e9; /* Light green background */
        border-radius: 10px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .header-title {
        color: #2e7d32; /* Darker green for title */
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .header-emoji {
        margin-right: 15px;
        font-size: 2.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: #33691e; /* Dark green for subtitle */
    }
    
    /* Button styling */
    .generate-button {
        background-color: #43a047 !important; /* Medium green button */
        color: white !important;
        font-weight: bold !important;
        padding: 0.75rem !important;
        border-radius: 4px !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
        margin: 1.5rem 0 !important;
    }
    
    .generate-button:hover {
        background-color: #2e7d32 !important; /* Darker green on hover */
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    /* Recipe title styling */
    .recipe-title {
        font-size: 2.8rem;
        color: #2c3e50;
        text-align: center;
        margin: 2rem 0 1rem 0;
        font-weight: bold;
    }
    
    /* Recipe metadata styling */
    .recipe-metadata {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0 2rem 0;
    }
    
    .recipe-metadata-item {
        display: flex;
        align-items: center;
        font-weight: bold;
        color: #555;
    }
    
    .recipe-metadata-emoji {
        margin-right: 0.5rem;
        font-size: 1.2rem;
    }
    
    /* Input field styling */
    .stTextInput label {
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    /* Override Streamlit's button styling */
    .stButton > button {
        background-color: #43a047 !important; /* Medium green button */
        color: white !important; 
    }
    
    .stButton > button:hover {
        background-color: #2e7d32 !important; /* Darker green on hover */
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header that matches the screenshot
    st.markdown("""
    <div class="header-container">
        <div class="header-title">
            <span class="header-emoji">🔍</span>
            Smart Recipe Generator
        </div>
        <div class="header-subtitle">
            Create personalized recipes based on your preferences!
        </div>
    </div>
    """, unsafe_allow_html=True)

    # User input
    user_input = st.text_input(
        "🤔 What kind of recipe would you like me to create?", 
        placeholder="e.g. I want to make yogurt",
        label_visibility="visible"
    )

    # Generate button styled like the screenshot
    generate_button = st.button("✏️ Generate Recipe", use_container_width=True)

    if generate_button and user_input:
        # Generate recipe section
        if generator_loaded:
            with st.spinner("Creating your personalized recipe..."):
                try:
                    # Generate the recipe
                    generated_recipe = generate_response(user_input, "./recipes.json")
                    
                    # Extract title (basic pattern matching)
                    title_match = re.search(r'(?:Recipe Title:|Title:)\s*([^\n]+)', generated_recipe, re.IGNORECASE)
                    title = title_match.group(1).strip() if title_match else "Personalized Recipe"
                    
                    # Add stars to title like in screenshot
                    title = f"** {title}"
                    
                    # Estimate cooking time
                    cooking_time = random.randint(20, 45)
                    
                    # Display the title without metadata
                    st.markdown(f'<div class="recipe-title">🍽️ {title}</div>', unsafe_allow_html=True)
                    
                    # Display the rest of the recipe
                    st.markdown(generated_recipe)
                    
                    # Add download option
                    st.download_button(
                        label="📥 Download Recipe",
                        data=generated_recipe,
                        file_name=f"recipe_{user_input.replace(' ', '_')[:20]}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Could not generate recipe: {str(e)}")
                    import traceback
                    print(f"Recipe generation error: {traceback.format_exc()}")
        else:
            st.warning("Recipe generator is not available. Please try again later.")