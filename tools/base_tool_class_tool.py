from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

# arg schema using pydantic


class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b


multiply_tool = MultiplyTool()

result = multiply_tool.invoke({"a": 3, "b": 3})

print("Tool Name: ", multiply_tool.name)
print("Tool Description: ", multiply_tool.description)
print("Tool args: ", multiply_tool.args)
print("Result: ", result)
print("Schema: ", multiply_tool.args_schema.model_json_schema())
