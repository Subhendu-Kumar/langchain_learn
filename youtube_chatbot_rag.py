from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

gemini = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)


def get_youtube_transcript(video_id: str):  # only the ID, not full URL
    try:
        # If you don’t care which language, this returns the “best” one
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=["en", "en-US"])
        # Flatten it to plain text
        transcript = " ".join(snippet.text for snippet in transcript_list)
        return transcript

    except TranscriptsDisabled:
        print("No captions available for this video.")


transcript = get_youtube_transcript("Z8Qip0kgl3A")

chunks = splitter.create_documents([transcript])

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
    collection_name="test",
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=["context", "question"],
)

# question = "What is the main topic of the video?"

# retrieved_docs = retriever.invoke(question)

# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# final_prompt = prompt.invoke({"context": context_text, "question": question})

# answer = gemini.invoke(final_prompt)

# print(answer.content)


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

parser = StrOutputParser()

main_chain = parallel_chain | prompt | gemini | parser

result = main_chain.invoke("What is the main topic of the video?")

print(result)
