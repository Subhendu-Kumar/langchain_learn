from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import Document

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"},
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"},
    ),
]

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="my_chroma_db",
    collection_name="sample",
)

vector_store.add_documents(docs)

# vector_store.get(include=["embeddings", "documents", "metadatas"])

data = vector_store.similarity_search(query="Who among these are a bowler?", k=2)

print(data)
