from langchain.tools import tool
from langchain.agents import create_agent
from datetime import datetime
from langchain.chat_models import init_chat_model
import os
from langchain.messages import HumanMessage

@tool
def get_current_time(city: str) -> str:
    """Get the current time in a given city"""
    return f"The current time in {city} is {datetime.now().strftime('%H:%M:%S')}"

@tool
def get_current_weather(city: str) -> str:
    """Get the current weather in a given city"""
    return f"The current weather in {city} is sunny, 42℃"

model = init_chat_model(
    "gemini-3-flash-preview-free",
    model_provider="openai",
    base_url="https://aihubmix.com/v1",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    temperature=0.7,
)

agent = create_agent(model=model, tools=[get_current_time, get_current_weather])

result = agent.invoke(
    {"messages": [HumanMessage("What is the current time in Beijing?")]}
)
print(result["messages"][-1].content)