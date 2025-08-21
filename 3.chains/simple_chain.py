from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"],
)

gemini = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

parser = StrOutputParser()

chain = prompt | gemini | parser

print(chain.invoke({"topic": "Python programming language"}))
