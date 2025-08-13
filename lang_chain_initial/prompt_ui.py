from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

gemini = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

st.header("Research Tool")
user_input = st.text_input("Enter your research question here:")

if st.button("Summarize"):
    if user_input:
        result = gemini.invoke(user_input)
        st.write("Result:", result.content)
    else:
        st.write("Please enter a question to summarize.")
