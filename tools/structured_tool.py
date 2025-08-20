from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number")
    b: int = Field(required=True, description="The second number")


def multiply_func(a: int, b: int) -> int:
    return a * b


multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput,
)

result = multiply_tool.invoke({"a": 3, "b": 3})

print("Tool Name: ", multiply_tool.name)
print("Tool Description: ", multiply_tool.description)
print("Tool args: ", multiply_tool.args)
print("Result: ", result)
print("Schema: ", multiply_tool.args_schema.model_json_schema())
