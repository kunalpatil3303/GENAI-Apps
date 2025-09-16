import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant. Please respond to the questions asked.'),
    ('user', 'Question:{question}')
])

# Streamlit UI
st.title("LangChain Demo with Mistral")
input_text = st.text_input("Enter your question here")

# Initialize model and parser
llm = Ollama(model="mistral", temperature=0)
output_parser = StrOutputParser()

# Generate response
if input_text:
    with st.spinner("Thinking..."):
        try:
            # Format the prompt
            formatted_prompt = prompt.format(question=input_text)
            # ✅ Use invoke() instead of __call__
            response = llm.invoke(formatted_prompt)
            # Parse output
            parsed_response = output_parser.parse(response)
            st.write("💬 Response:")
            st.write(parsed_response)
        except Exception as e:
            st.error(f"Error generating response: {e}")
else:
    st.info("👆 Please enter a question above")
