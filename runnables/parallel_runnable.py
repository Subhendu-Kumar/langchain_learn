from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a Linkedin post about {topic}", input_variables=["topic"]
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1, model, parser),
        "linkedin": RunnableSequence(prompt2, model, parser),
    }
)

result = parallel_chain.invoke({"topic": "AI"})

print("twitter: \n\n", result["tweet"])
print("\n\n\n\n")
print("linkedin: \n\n", result["linkedin"])
