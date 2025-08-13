from langchain.schema.runnable import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

gemini = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

prompt_1 = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

prompt_2 = PromptTemplate(
    template="Explain the following joke - {text}", input_variables=["text"]
)

parser = StrOutputParser()

chain = RunnableSequence(prompt_1, gemini, parser, prompt_2, gemini, parser)

print(chain.invoke({"topic": "AI"}))
