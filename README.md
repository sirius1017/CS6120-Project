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
pip install -U langchain-chroma
pip install -q -U langchain-google-genai


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
We could test "Out of Vocabulary" (0OV) words, though we're more interested in the ML parts of youralgorithms. For example, for cooking recipes, l could type in something like:
1. "Can you find me a recipe that does not have cheries and berries?" (This may be somewhat challenging
because of the negation.)
2. "What's a gluten-free recipe for pies that uses bread and meat?" (Your LLM may not be able to answer
this)
3. "l've seen a recipe calls for cooking the milk, cream, and sugar until the sugar has dissolved. Then, we wouldmix with a cup, while adding vanila extract, We need an ice cream maker for churning according to themanufacturer's directions, but l don't know if l have it. We would need to serve immediately or ripen in thefreezer, Do you know what this recipe is for?" (Somewhat ofa long guery.)
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
## retriever
处理query部分，让查找到的embeddings更准确
## generator
prompt(可以用LangChain prompt hub https://python.langchain.com/docs/tutorials/rag/#orchestration)
试一下不同的模型
## 异常数据处理问题
例如chocolate被标注为"candies, semisweet chocolate"导致无法匹配。是否可以直接修改数据集？
"spices, pepper, black"
## 数值调整
根据nutrition排序的时候，retrieve_full_recipes的top_k值设置为多少比较合适？
## 其他
考虑一些edge cases queries
evaluation
可以和膳食指南联系一下
能否通过厨具-->preparation和菜品名-->title找到可以使用的菜谱？
## ddl April 24

## 其他提示
in your README.md in Github, providing the provenance of how you created your capability with respect to datainclude two aspects, which are part of the grading rubric:
1. Reproducibility - lf someone wanted to, wouuld they be able to re-create your LLM? it's Ok if the data isn'topen source, and guite frankly, l would encourage it to the extent that we can verify your solutions. lf it'sclosed source and proprietary, please let us know how we can verify -- which leads me to the second
criterion.
2. Verifiability - Citing the appropriate passages and source material (in any return messages or outside of itwill provide confidence in the the text that it returns. lf you've not included the data in the repository, that istotally fine. Please have it available when demonstrating your work.


# Data source
https://www.reddit.com/r/Cooking/comments/2b4vt4/cookbook_download_mastering_the_art_of_french/
https://www.klimareporter.de/images/dokumente/2023/03/3a84daaa-7c0b-4fcf-84c8-85aaa63683c4.pdf
