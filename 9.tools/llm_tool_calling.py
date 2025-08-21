from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers a and b this tool returns their product"""
    return a * b


gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

llm_with_tools = gemini.bind_tools([multiply])

query = HumanMessage("can you multiply 3 with 1000")

messages = [query]

result = llm_with_tools.invoke(messages)

messages.append(result)

tool_result = multiply.invoke(result.tool_calls[0])

messages.append(tool_result)

print(llm_with_tools.invoke(messages).content)
