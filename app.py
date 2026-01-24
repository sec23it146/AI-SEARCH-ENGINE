import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

# 🔑 Load API keys from Hugging Face Secrets
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# 🤖 Load LLM
llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.7,
    groq_api_key=groq_api_key
)

# 🔍 Web Search Tool
search = TavilySearch(
    max_results=5,
    tavily_api_key=tavily_api_key
)

# 🧠 Agent Logic
def agent_chat(user_input, history):
    # 1️⃣ Search the web
    res = search.invoke(user_input)

    # 2️⃣ Collect web content
    context = "\n".join(
        [r["content"] for r in res.get("results", [])]
    )

    # 3️⃣ Prompt
    prompt = f"""
You are Rahins AI Search Agent.
Use the web information below to answer clearly.

Web Information:
{context}

User Question:
{user_input}

Final Answer:
"""

    # 4️⃣ Ask LLM
    response = llm.invoke(prompt)

    return response.content.strip()

# 🎨 Gradio UI
gr.ChatInterface(
    fn=agent_chat,
    title="Rahins AI Search Agent",
    description="Ask anything. This AI searches the web and answers intelligently."
).launch()
