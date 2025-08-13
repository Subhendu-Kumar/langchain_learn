from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

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
    template="Generate 5 short mcq from the following text: \n {text}",
    input_variables=["text"],
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

parser = StrOutputParser()

classifier_chain = prompt_1 | model | parser_0

result = classifier_chain.invoke({"feedback": "this product is the worst product"})

print("Sentiment Classification:", result.sentiment)
