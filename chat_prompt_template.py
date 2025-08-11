from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# gemini = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

chat_template = ChatPromptTemplate(
    [
        ("system", "you are a helpful {domain} expert"),
        ("human", "Explain in simple terms. what is {topic} ?"),
    ]
)

prompt = chat_template.invoke({"domain": "science", "topic": "quantum mechanics"})

print(prompt)
