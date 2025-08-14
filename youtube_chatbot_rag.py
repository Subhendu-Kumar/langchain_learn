from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


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


transcript = get_youtube_transcript("Gfr50f6ZBvo")

chunks = splitter.create_documents([transcript])

print(chunks)

# vector_store = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     persist_directory="my_chroma_db",
#     collection_name="sample_test",
# )

# print(vector_store.index_to_docstore_id)
