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

    model = os.getenv(
        "HF_MODEL",
        "Qwen/Qwen2.5-7B-Instruct",
    )

    provider = os.getenv(
        "HF_PROVIDER",
        "featherless-ai",
    )

    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        provider=provider,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        huggingfacehub_api_token=hf_token,
    )

    return ChatHuggingFace(llm=endpoint)