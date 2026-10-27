
from openai import OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# Vector Embeddings
client = OpenAI()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name=["newdata", "machin_learing"],
    embedding=embedding_model
)

# Take user input
user_query = input("Ask something -->> ")

search_results = vector_db.similarity_search(query=user_query)

context = "\n\n\n".join([
    f"Page Content: {result.page_content}\n"
    f"Page Number: {result.metadata['page_label']}\n"
    f"File Location: {result.metadata['source']}"
    for result in search_results
])

system_prompt = f"""
You are a helpful AI assistant who answers user queries based on the
available context retrieved from a PDF file, including page content
and page numbers.

Context:
{context}
"""

response = client.chat.completions.create(
    model="gemini-3.5-flash-lite",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
)

print(f"-->> {response.choices[0].message.content}")
