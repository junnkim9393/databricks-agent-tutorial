import operator
import os
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langchain_databricks import ChatDatabricks
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def build_agent(tools: list):
    llm = ChatDatabricks(
        endpoint=os.environ.get("MODEL_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
    )
    llm_with_tools = llm.bind_tools(tools)

    system = SystemMessage(content="You are a helpful assistant. Only call tools when the user's request clearly requires them. If no tool is relevant, answer directly from your knowledge.")

    def call_model(state):
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    def should_continue(state):
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()
