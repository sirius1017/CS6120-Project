from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# 如果你用的是新版 huggingface
from langchain_huggingface import HuggingFaceEmbeddings

# 加载已有的 Chroma 向量数据库
persist_directory = "./chroma_db"

# 创建 embedding 对象（必须和写入时一致）
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 128}
)

# 加载 Chroma 数据库
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# ✅ 检查存储的文档数量
print("📦 Stored document count:", vectorstore._collection.count())

# ✅ 测试一个简单的查询
query = "How to make healthy yogurt parfait?"
results = vectorstore.similarity_search(query, k=3)

# ✅ 打印结果内容和元数据
print("\n🔍 Top 3 results:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print("📄 Content:\n", doc.page_content[:300], "...")
    print("🔖 Metadata:", doc.metadata)
