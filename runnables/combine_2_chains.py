import nakli_llm_v2

llm = nakli_llm_v2.NakliLLM()

parser = nakli_llm_v2.NakliStrOutputParser()

prompt_1 = nakli_llm_v2.NakliPromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

prompt_2 = nakli_llm_v2.NakliPromptTemplate(
    template="explain the following joke {response}", input_variables=["response"]
)

chain_1 = nakli_llm_v2.RunnableConnector([prompt_1, llm])

chain_2 = nakli_llm_v2.RunnableConnector([prompt_2, llm, parser])

final_chain = nakli_llm_v2.RunnableConnector([chain_1, chain_2])

result = final_chain.invoke({"topic": "AI"})

print("result : " + result)
