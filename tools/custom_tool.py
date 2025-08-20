from langchain_core.tools import tool

# Step 1 - create a function

# def multiply(a, b):
#     """Multiply two numbers"""
#     return a * b

# Step 2 - add type hints

# def multiply(a: int, b:int) -> int:
#     """Multiply two numbers"""
#     return a*b

# Step 3 - add tool decorator


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


result = multiply.invoke({"a": 3, "b": 5})

print("Tool Name: ", multiply.name)
print("Tool Description: ", multiply.description)
print("Tool args: ", multiply.args)
print("Result: ", result)
print("Schema: ", multiply.args_schema.model_json_schema())
