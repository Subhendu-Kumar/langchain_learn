from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

docs = [
    Document(
        page_content=(
            """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
        ),
        metadata={"source": "Doc1"},
    ),
    Document(
        page_content=(
            """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
        ),
        metadata={"source": "Doc2"},
    ),
    Document(
        page_content=(
            """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
        ),
        metadata={"source": "Doc3"},
    ),
    Document(
        page_content=(
            """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
        ),
        metadata={"source": "Doc4"},
    ),
]

vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

compressor = LLMChainExtractor.from_llm(
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro")
)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever, base_compressor=compressor
)

query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
