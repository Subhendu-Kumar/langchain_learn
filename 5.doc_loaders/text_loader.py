from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

loader = TextLoader("cricket.txt", encoding="utf-8")
doc = loader.load()

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Summarize the following poem in 50 words: \n\n {text}",
    input_variables=["text"],
)

chain = prompt | model | parser

result = chain.invoke({"text": doc[0].page_content})

print(result)
