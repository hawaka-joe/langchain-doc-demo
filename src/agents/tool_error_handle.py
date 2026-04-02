from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import tool

@tool
def read_data(data: str) -> str:
    """Read the data"""
    return f"Read results for {data}"

@tool
def write_data(data: str) -> str:
    """Write the data"""
    return f"Write results for {data}"

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gemini-3-flash-preview-free",
    tools=[read_data, write_data],
    middleware=[handle_tool_errors]
)