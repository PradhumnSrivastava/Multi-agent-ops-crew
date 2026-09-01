import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()


def create_llm(
    max_new_tokens: int = 500,
    temperature: float = 0.1,
) -> ChatHuggingFace:
    """Create the configured LLM used by the application."""

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN is not configured.")

    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        provider="featherless-ai",
        task="text-generation",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        huggingfacehub_api_token=hf_token,
    )

    return ChatHuggingFace(llm=endpoint)