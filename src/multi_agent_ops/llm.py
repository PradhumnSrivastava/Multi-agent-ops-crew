import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()


def create_llm() -> ChatHuggingFace:
    """Create the shared LLM used by all agents."""

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

    max_new_tokens = int(
        os.getenv("HF_MAX_NEW_TOKENS", "500")
    )

    temperature = float(
        os.getenv("HF_TEMPERATURE", "0.1")
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