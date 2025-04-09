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


# tests
## basic recipes
How to make a Chocolate Frosting
How to make vanilla cupcakes?
How to make Yogurt Parfaits？
Give me a recipe for matcha sponge cake.
How do I bake gluten-free banana bread?
Tell me how to make a moist carrot cake.
What's a simple brownie recipe with no eggs?

## creative/fusion recipes
Give me a recipe of peppercorn chocolate cake.
Give me a recipe of Sichuan chocolate cake.
How can I make Japanese-style chili tofu curry?
Make me a Mediterranean pizza with za'atar and labneh.
How to cook a fusion of tiramisu and mango mousse?

## restricted ingredients
Give me a cake recipe using only flour, eggs, and honey.
I have matcha powder, rice flour, and coconut milk. What can I bake?
Make something with strawberries, granola, and yogurt.
Suggest a dessert using tofu, sesame, and brown sugar.

## edge cases
1. 空输入 / 极短查询
help
cake
???
2. 拼写错误 / 模糊语义
give me a recipy for chiken curryy  
how to mak chocolet frostin  
what is the best way to bak a piza
3. 语义不明确 / 非命令式问题
I like spicy food  
Too much sugar  
Orange? 
4. 要求不存在的配方 / 创意料理
How to make a sushi lasagna?  
Give me a Sichuan pepper tiramisu  
Can I make gluten-free bacon brownies with tofu?  
5. 查询与数据完全不匹配
How do I fix a broken dishwasher with granola?  
Give me a recipe to prepare quantum foam tart  
How to knit a cake  
6. 包含限定条件/过滤条件的 query
Give me a low-fat vegan chocolate dessert  
Show me a high-protein breakfast with no eggs  
What is a cake recipe under 200 calories per serving?  
7. 多语言输入（如果支持）
请给我一个四川辣椒蛋糕的做法  
Receta para un pastel sin azúcar ni harina  
Comment faire un gâteau au poivre du Sichuan  


# 改进
## index部分
data_preprocessing.py
    修改ingest_to_chroma(threshold=100)中的threshold
## retriever
处理query部分，让查找到的embeddings更准确
## generator
prompt(可以用LangChain prompt hub https://python.langchain.com/docs/tutorials/rag/#orchestration)
试一下不同的模型
## 其他
考虑一些edge cases queries
evaluation
可以和膳食指南联系一下
## ddl April 24
