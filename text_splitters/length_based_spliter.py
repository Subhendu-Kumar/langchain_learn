from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("curriculum.pdf")

docs = loader.load()

text = """
    t’s important to remember the difference between encoding and encrypting. Encoding is the process of making the sent message as identical to the original as possible to minimize errors. Encrypting is making the message unreadable to everyone but the intended recipient. There are several different kinds of encryption, the main ones being asymmetric and symmetric. We covered the former in this article, and explained that in contrast with symmetric encryption, asymmetric encryption introduces a public and private key rather than just one private key shared between the parties, making it possible to securely communicate across insecure channels. The advantage of asymmetric encryption becomes obvious in cryptocurrency, where the public key is used to receive funds and check balance and transactions, whereas the private key is the only way to actually sign messages and send the tokens.
"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separator="",
)

# result = splitter.split_text(text)

result = splitter.split_documents(docs)

for line in result:
    print(line.page_content)
    print("\n\n")
