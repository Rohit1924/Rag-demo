
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent / "rohit1.pdf"

# Load the PDF file
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()
# print(docs[18])

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)
text = text_splitter.split_documents(documents=docs)

# Generate vector embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_store = QdrantVectorStore.from_documents(
    documents=text,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="newdata"
)

print("Document indexing completed successfully.")
