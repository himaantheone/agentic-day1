from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage
load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano", seed=6)

resp1 = llm.invoke("We are building an AI system for processing medical insurance claims.")

resp2 = llm.invoke("What are the main risks in this system?")

messages = [
    SystemMessage(content="You are a senior AI architect reviewing production systems."),
    HumanMessage(content="We are building an AI system for processing medical insurance claims."),
    HumanMessage(content="What are the main risks in this system?")
]

response = llm.invoke(messages)
print(response)

"""
Reflection:

1. Why did string-based invocation fail?
   Because LLm calls are stateless, it doesnt know the context of previous invokes.

2. Why does message-based invocation work?
   It works because the it provides context of conversation history and using that llm can give a grounded output

3. What would break in a production AI system if we ignore message history?
   The llm will hallucinate and answer relevancy to the question asked will be not accurate

"""