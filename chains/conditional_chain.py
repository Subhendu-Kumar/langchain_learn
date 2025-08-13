from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch, RunnableLambda

load_dotenv()


class FeedBack(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback text"
    )


parser_0 = PydanticOutputParser(pydantic_object=FeedBack)

prompt_1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative: \n\n {feedback} \n\n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser_0.get_format_instructions()},
)

prompt_2 = PromptTemplate(
    template="write an appropiate response to this positive feedback: \n {feedback}",
    input_variables=["feedback"],
)
prompt_3 = PromptTemplate(
    template="write an appropiate response to this negative feedback: \n {feedback}",
    input_variables=["feedback"],
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

parser = StrOutputParser()

classifier_chain = prompt_1 | model | parser_0

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt_2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt_3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment"),
)


chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "this product is the worst product"})

print(result)