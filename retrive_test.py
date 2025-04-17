
from retriever import retrieve_full_recipes, top_k_by_nutrient

if __name__=="__main__":
    
    # retriver用法示例

    print("===================similarity search for title===================")
    # 通过title搜索近似（eg 菜品名称，营养特色）
    print("5 gluten-free cake recipes")
    retrieve_res = retrieve_full_recipes(query= "gluten free cake",
                          mode= "title",
                          json_path= "./recipes.json")
    # print(json.dumps(retrieve_res, indent=2, ensure_ascii=False))
    for i, r in enumerate(retrieve_res, start=1):
        title = r.get("title", "N/A")
        raw_ings = r.get("ingredients", [])
        ingredients = [i["text"].split(",")[0].lower() for i in raw_ings if "text" in i and i["text"]]
        print(f"\n[{i}] {title}")
        print("Ingredients:", ", ".join(ingredients))
    print()


    print("===================similarity search for ingredients===================")
    # 通过ingredient搜索原材料最接近的菜谱
    print("5 recipes using flour egg and milk")
    retrieve_res = retrieve_full_recipes(query= "egg;flour;milk", #注意：需要按照字母表排序，并用;分隔
                          mode= "ingredient",
                          json_path= "./recipes.json")
    # print(json.dumps(retrieve_res, indent=2, ensure_ascii=False))
    for i, r in enumerate(retrieve_res, start=1):
        title = r.get("title", "N/A")
        raw_ings = r.get("ingredients", [])
        ingredients = [i["text"].split(",")[0].lower() for i in raw_ings if "text" in i and i["text"]]
        print(f"\n[{i}] {title}")
        print("Ingredients:", ", ".join(ingredients))

    print()
    print("===============similarity search for ingredients&nutrition===============")
    # 通过ingredient及nutr_values_per100g从原材料最接近的菜谱中挑选特定营养成分的菜谱
    # （eg.从N个使用cocoa和flour的食谱里选出热量最低的5个食谱）
    print("5 low cal food with cocoa and flour")
    retrieved = retrieve_full_recipes(query="cocoa;flour", 
                                      mode="ingredient", 
                                      json_path="./recipes.json", 
                                      top_k=10) # 提升此处top_k可能会导致similarity过低的问题
    ranked = top_k_by_nutrient(retrieved, 
                               nutrient="energy", # "energy""fat""protein""salt""saturates""sugars"
                               k=5, 
                               descending = False) # 选出最低的5个)
    for i, r in enumerate(ranked, start=1):
        title = r.get("title", "N/A")
        raw_ings = r.get("ingredients", [])
        ingredients = [i["text"].split(",")[0].lower() for i in raw_ings if "text" in i and i["text"]]
        energy = r.get("nutr_values_per100g", {}).get("energy", "N/A")
        print(f"\n[{i}] {title}")
        print("Ingredients:", ", ".join(ingredients))
        print("Energy (per 100g):", energy)

    print("")
    # （eg.从N个使用chickpeas的食谱里选出最高蛋白的5个食谱）
    print("5 high protein food with chickpeas")
    retrieved = retrieve_full_recipes(query="chickpeas", 
                                      mode="ingredient", 
                                      json_path="./recipes.json", 
                                      top_k=10)
    ranked = top_k_by_nutrient(retrieved, 
                               nutrient="protein", # "energy""fat""protein""salt""saturates""sugars"
                               k=5, 
                               descending = True) # 选出最高的5个)
    for i, r in enumerate(ranked, start=1):
        title = r.get("title", "N/A")
        raw_ings = r.get("ingredients", [])
        ingredients = [i["text"].split(",")[0].lower() for i in raw_ings if "text" in i and i["text"]]
        energy = r.get("nutr_values_per100g", {}).get("protein", "N/A")
        print(f"\n[{i}] {title}")
        print("Ingredients:", ", ".join(ingredients))
        print("protein (per 100g):", energy)
