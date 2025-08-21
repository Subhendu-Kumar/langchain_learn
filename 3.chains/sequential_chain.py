from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

prompt_1 = PromptTemplate(
    template="Generate detailed report on {topic}",
    input_variables=["topic"],
)

prompt_2 = PromptTemplate(
    template="Generate summary of key points from the report: {text}",
    input_variables=["text"],
)

gemini = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

parser = StrOutputParser()

chain = prompt_1 | gemini | parser | prompt_2 | gemini | parser

result = chain.invoke({"topic": "biology"})

print("Summary:", result)
