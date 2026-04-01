from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
import os

base_model = init_chat_model(
    "gemini-3-flash-preview-free",
    model_provider="openai",
    base_url="https://aihubmix.com/v1",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    temperature=0.7,
)

advanced_model = init_chat_model(
    "glm-4.7-flash-free",
    model_provider="openai",
    base_url="https://aihubmix.com/v1",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    temperature=0.7,
)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_len = len(request.state["messages"][0].content)
    print(request.state["messages"][0].content)
    print(f"Message length: {message_len}")
    
    if message_len < 20:
        model = base_model
    else:
        model = advanced_model
        
    return handler(request.override(model=model))

agent = create_agent(model=base_model, middleware=[dynamic_model_selection])

result = agent.invoke(
    {"messages": [HumanMessage("I am zhouhuiwei, come from China ,who are you?")]}
)
print(result["messages"][-1].content)