import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN is not configured in .env")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.1,
    huggingfacehub_api_token=hf_token,
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke(
    "Explain in one sentence what a multi-agent AI system is."
)

print(response.content)