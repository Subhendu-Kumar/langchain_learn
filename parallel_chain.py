from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema.runnable import RunnableParallel

load_dotenv()

prompt_1 = PromptTemplate(
    template="Generate short and simple notes on following text: \n {text}",
    input_variables=["text"],
)

prompt_2 = PromptTemplate(
    template="Generate 5 short mcq from the following text: \n {text}",
    input_variables=["text"],
)

prompt_3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single doc: \n {notes} and {quiz}",
    input_variables=["notes", "quiz"],
)

model_1 = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

model_2 = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
)

parser = StrOutputParser()


parallel_chain = RunnableParallel(
    {"notes": prompt_1 | model_1 | parser, "quiz": prompt_2 | model_2 | parser}
)

merge_chain = prompt_3 | model_1 | parser

final_chain = parallel_chain | merge_chain

text = """
Biology is the scientific study of life and living organisms. It is a broad natural science that encompasses a wide range of fields and unifying principles that explain the structure, function, growth, origin, evolution, and distribution of life. Central to biology are five fundamental themes: the cell as the basic unit of life, genes and heredity as the basis of inheritance, evolution as the driver of biological diversity, energy transformation for sustaining life processes, and the maintenance of internal stability (homeostasis).[1][2]

Biology examines life across multiple levels of organization, from molecules and cells to organisms, populations, and ecosystems. Subdisciplines include molecular biology, physiology, ecology, evolutionary biology, developmental biology, and systematics, among others. Each of these fields applies a range of methods to investigate biological phenomena, including observation, experimentation, and mathematical modeling. Modern biology is grounded in the theory of evolution by natural selection, first articulated by Charles Darwin, and in the molecular understanding of genes encoded in DNA. The discovery of the structure of DNA and advances in molecular genetics have transformed many areas of biology, leading to applications in medicine, agriculture, biotechnology, and environmental science.

Life on Earth is believed to have originated over 3.7 billion years ago.[3] Today, it includes a vast diversity of organisms—from single-celled archaea and bacteria to complex multicellular plants, fungi, and animals. Biologists classify organisms based on shared characteristics and evolutionary relationships, using taxonomic and phylogenetic frameworks. These organisms interact with each other and with their environments in ecosystems, where they play roles in energy flow and nutrient cycling. As a constantly evolving field, biology incorporates new discoveries and technologies that enhance the understanding of life and its processes, while contributing to solutions for challenges such as disease, climate change, and biodiversity loss. 
"""

result = final_chain.invoke({"text": text})

print("Summary:", result)

# final_chain.get_graph().print_ascii()
