import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from elevenlabs.client import ElevenLabs
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

gemini = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

elevenlabs = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

audio_url = "Recording.m4a"
audio_file = open(audio_url, "rb")

transcription = elevenlabs.speech_to_text.convert(
    file=audio_file,
    model_id="scribe_v1",  # Model to use, for now only "scribe_v1" is supported
    tag_audio_events=True,  # Tag audio events like laughter, applause, etc.
    language_code="eng",  # Language of the audio file. If set to None, the model will detect the language automatically.
    diarize=True,  # Whether to annotate who is speaking
)

print(transcription.text)

question = (
    "What is the difference between a process and a thread in an operating system?"
)

base_answer = """A process is an independent program in execution with its own memory space, resources, and scheduling.

A thread is a lightweight unit of execution within a process that shares the same memory and resources of the process but has its own execution path (program counter, stack, and registers).

Threads are faster to create and switch between compared to processes because they avoid the overhead of separate memory allocation."""

# user_answer = "Process is like a full program with its own memory, and thread is smaller, inside a process, sharing the same memory. Process is heavy, thread is light and faster."


class OutputFormat(BaseModel):
    score: int = Field(
        description="score of the ans provided by the user in comparision with base ans provided, score between 0 to 100"
    )
    feedback: str = Field(
        description="Give a detailed feed back to the user according to the comparision between the ans provided by the user and the base ans. also in feedback dont include word like base ans"
    )


parser = PydanticOutputParser(pydantic_object=OutputFormat)

prompt = PromptTemplate(
    template="""
        you are examiner, and you are very good at checking user provided ans in comparision with the base ans provided.

        question: {question} \n\n
        base answer: {base_answer} \n\n
        user answer: {user_answer} \n\n
        {format_instruction}
    """,
    input_variables=["question", "base_answer", "user_answer"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

print("initiarted chain")
chain = prompt | gemini | parser

result = chain.invoke(
    {
        "question": question,
        "base_answer": base_answer,
        "user_answer": transcription.text,
    }
)

print(result)
