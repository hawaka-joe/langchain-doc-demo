from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
import os
from pprint import pprint

model = init_chat_model(
    "deepseek-chat",
    model_provider="openai",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

@tool
def get_weather(city: str) -> str:
    """Get the current weather in a given city"""
    return f"The current weather in {city} is sunny, 42℃"

agent = create_agent(model=model, tools=[get_weather])

result = agent.invoke(
    {"messages": [HumanMessage("What is the current weather in Beijing?")]}
)
print(result["messages"][-1].content)
pprint(result)

"""
{
    "messages": [
        {
            "type": "HumanMessage",
            "content": "What is the current weather in Beijing?",
            "id": "b80ea662-8032-4134-bdbb-ef87083ecbe0"
        },
        {
            "type": "AIMessage",
            "content": "I'll check the current weather in Beijing for you.",
            "tool_calls": [
                {
                    "name": "get_weather",
                    "args": {
                        "city": "Beijing"
                    },
                    "id": "call_00_WAVomKtSu9jzrbbozDEdSX9e"
                }
            ],
            "model": "deepseek-chat",
            "finish_reason": "tool_calls",
            "usage": {
                "prompt_tokens": 308,
                "completion_tokens": 54,
                "total_tokens": 362
            },
            "id": "lc_run--019d5172-f52e-74f1-811a-d607441443c2-0"
        },
        {
            "type": "ToolMessage",
            "name": "get_weather",
            "content": "The current weather in Beijing is sunny, 42℃",
            "tool_call_id": "call_00_WAVomKtSu9jzrbbozDEdSX9e",
            "id": "6ed7dda7-c79b-452d-b3a1-5a36f0f4d034"
        },
        {
            "type": "AIMessage",
            "content": "The current weather in Beijing is sunny with a temperature of 42°C.",
            "model": "deepseek-chat",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 389,
                "completion_tokens": 16,
                "total_tokens": 405
            },
            "id": "lc_run--019d5172-fe66-7140-b290-0b65c4da30fb-0"
        }
    ]
}
"""