import os
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model(
    "gemini-3-flash-preview-free",
    model_provider="openai",
    base_url="https://aihubmix.com/v1",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    temperature=0.7,
)

# invoke way 1
result = model.invoke("how to be rich?")
print(result.content)

# invoke way 2
conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]
response = model.invoke(conversation)
print(response)

# invoke way 3
response = model.invoke([
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate: I love programming."),
    AIMessage(content="J'adore la programmation."),
    HumanMessage(content="Translate: I love building applications.")
])
print(response)