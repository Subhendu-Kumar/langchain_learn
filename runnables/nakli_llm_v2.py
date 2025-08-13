import random
from abc import ABC, abstractmethod


class Runnable(ABC):
    @abstractmethod
    def invoke(input_dict):
        pass


class NakliLLM(Runnable):
    def __init__(self):
        self.response_list = [
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence",
        ]

    def invoke(self, prompt):
        return {"response": random.choice(self.response_list)}

    def predict(self, prompt):
        return {"response": random.choice(self.response_list)}


class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)


class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data


class NakliStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data["response"]


template = NakliPromptTemplate(
    template="Write a {length} poem about {topic}", input_variables=["length", "topic"]
)

llm = NakliLLM()

parser = NakliStrOutputParser()

chain = RunnableConnector([template, llm, parser])

result = chain.invoke({"length": "short", "topic": "india"})

# print(result)
