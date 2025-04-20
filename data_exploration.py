import json, re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Load the data
with open('recipes.json') as f:
    data = json.load(f)

# Check number of records
print(f"Total recipes: {len(data)}")

# Inspect the keys of the first recipe
first_recipe = data[0]
print("Top-level keys:", first_recipe.keys())

# Flatten into a list of dicts
records = []
for item in data:
    record = {
        "id": item.get("id"),
        "title": item.get("title"),
        "url": item.get("url"),
        "partition": item.get("partition"),
        "fsa_lights_per100g": item.get("fsa_lights_per100g", {}),
        "num_ingredients": len(item.get("ingredients", [])),
        "ingredients": [i["text"] for i in item.get("ingredients", [])],
        "instructions": [i["text"] for i in item.get("instructions", [])],
        "nutrition_per100g": item.get("nutr_values_per100g", {}),
        "energy": item["nutr_values_per100g"].get("energy"),
        "fat": item["nutr_values_per100g"].get("fat"),
        "protein": item["nutr_values_per100g"].get("protein"),
        "salt": item["nutr_values_per100g"].get("salt"),
        "saturates": item["nutr_values_per100g"].get("saturates"),
        "sugars": item["nutr_values_per100g"].get("sugars"),
    }
    records.append(record)

df = pd.DataFrame(records)

print(df.isnull().sum())

# Look for nulls
df['title'].sample(10)  

# Title
df["title_length"] = df["title"].apply(lambda x: len(x.split()))
df["title_length"].hist(bins=20)
plt.title("Recipe Title Length Distribution")
plt.xlabel("Number of Words in Title")
plt.ylabel("Frequency")
plt.savefig("title_length_distribution.png", dpi=300)
plt.show()

duplicate_titles = df["title"].duplicated().sum()
print(f"Duplicate titles found: {duplicate_titles}")

# Optionally preview some
df[df["title"].duplicated(keep=False)].sort_values("title").head(10)

# Frequent Title
df["title"].value_counts().head(20).plot(kind="barh")
plt.title("Most Common Recipe Titles")
plt.xlabel("Count")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top_titles.png", dpi=300)
plt.show()

# Ingredients
ingredient_counter = Counter()

for ingr_list in df["ingredients"]:
    ingredient_counter.update([i.lower().strip() for i in ingr_list])

top_ingredients = pd.Series(ingredient_counter).sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 6))
top_ingredients.plot(kind="barh")
plt.title("Top 20 Most Common Ingredients")
plt.xlabel("Frequency")
plt.gca().invert_yaxis()  # Most frequent at the top
plt.tight_layout()
plt.savefig("top_ingredients.png", dpi=300)
plt.show()

for ingr_list in df["ingredients"]:
    ingredient_counter.update([i.lower().strip() for i in ingr_list])

# Instructions & Cooking Method Exploration
# Common cooking actions
cooking_methods = [
    "bake", "boil", "fry", "mix", "steam", "grill", 
    "simmer", "sauté", "smoke", "stew", "roast", "broil", 
    "poach", "blanch", "microwave", "layer", "knead", "whisk"
]

method_counter = Counter()

for instr_list in df["instructions"]:
    for step in instr_list:
        words = re.findall(r"\b\w+\b", step.lower())
        for word in words:
            if word in cooking_methods:
                method_counter[word] += 1

# Sort and convert to Series for plotting
method_series = pd.Series(method_counter).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
method_series.plot(kind="bar")
plt.title("Common Cooking Methods")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("common_cooking_methods_distribution.png", dpi=300)
plt.show()


# Nutritional Value Exploration
df[["energy", "fat", "protein", "salt", "saturates", "sugars"]].hist(bins=30, figsize=(12, 8))

# Correlation
sns.heatmap(df[["energy", "fat", "protein", "sugars"]].corr(), annot=True)

plt.figure(figsize=(10, 6))
sns.heatmap(df[["energy", "fat", "salt", "saturates", "sugars", "protein"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Nutritional Values")
plt.savefig("correlation_nutrition.png", dpi=300)
plt.show()

# FSA Labels (fat, sugar, salt, etc.)
df["fsa_fat_rating"] = df["fsa_lights_per100g"].apply(lambda x: x.get("fat") if isinstance(x, dict) else None)
df["fsa_salt_rating"] = df["fsa_lights_per100g"].apply(lambda x: x.get("salt") if isinstance(x, dict) else None)
df["fsa_saturates_rating"] = df["fsa_lights_per100g"].apply(lambda x: x.get("saturates") if isinstance(x, dict) else None)
df["fsa_sugars_rating"] = df["fsa_lights_per100g"].apply(lambda x: x.get("sugars") if isinstance(x, dict) else None)

label_map = {"green": 0, "orange": 1, "red": 2}

for col in ["fsa_fat_rating", "fsa_salt_rating", "fsa_saturates_rating", "fsa_sugars_rating"]:
    df[col + "_score"] = df[col].map(label_map)


# Plot all distributions together
# Melt the ratings into one column
fsa_cols = ["fsa_fat_rating", "fsa_salt_rating", "fsa_saturates_rating", "fsa_sugars_rating"]
df_melted = df.melt(value_vars=fsa_cols, var_name="nutrient", value_name="rating")

# Plot all distributions together
plt.figure(figsize=(10, 6))
sns.countplot(data=df_melted, x="rating", hue="nutrient", order=["green", "orange", "red"])
plt.title("FSA Rating Distribution by Nutrient")
plt.ylabel("Count")
plt.xlabel("FSA Label")
plt.legend(title="Nutrient")
plt.tight_layout()
plt.savefig("fsa_rating_distribution_all.png", dpi=300)
plt.show()

print(df.describe())