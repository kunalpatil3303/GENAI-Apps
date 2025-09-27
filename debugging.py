from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

class State(TypedDict):
    messages:Annotated[list[BaseMessage], add_messages]


model = ChatOpenAI(model = "gpt-4o", temperature = 0)

def make_default_graph():
    graph_workflow = StateGraph(State)


    def call_model(state:State):
        return{"messages":[model.invoke(state["messages"])]}
    
    graph_workflow.add_node("agent", call_model)

    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("agent", END)

    agent = graph_workflow.compile()
    return agent

agent = make_default_graph()

