import os
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv

# --- Environment Variable Loading ---
# Call the function to actually load the variables from the .env file
load_dotenv()

# --- Model Initialization ---
# Best practice: Load API key from environment variables for security.
# This prevents you from accidentally sharing your key in your code.
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set!")

# Initialize the ChatGroq model with the API key and desired settings
model = ChatGroq(
    api_key=groq_api_key,
    model="gemma2-9b-it",
    temperature=0  # Set to 0 for deterministic, predictable translations
)

# --- LangChain Prompt and Chain Definition ---
# A clear, generic template for translation tasks.
# The `language` and `text` will be provided by the user.
generic_template = "Translate the following into {language}:"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
    ]
)

# The output parser simply converts the model's chat message output to a string.
parser = StrOutputParser()

# --- The Translation Chain ---
# This is the core logic: prompt -> model -> parser
chain = prompt | model | parser


# --- FastAPI Application Setup ---
app = FastAPI(
    title="LangServe with Groq Translation",
    version="1.0",
    description="A simple API for translating text using LangChain and Groq's Gemma2 model."
)

# Add the LangServe routes to the FastAPI app.
# The chain will be accessible at the `/groq` endpoint.
add_routes(
    app,
    chain,
    path="/groq"
)

# --- Main Entry Point ---
# This block allows you to run the server directly from the script
# using `python main.py`
if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://localhost:8000")
    uvicorn.run(app, host="localhost", port=8000)
