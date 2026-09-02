import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()


def create_llm(
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> ChatHuggingFace:
    """Create the shared Hugging Face chat model."""

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

    configured_max_tokens = int(
        os.getenv("HF_MAX_NEW_TOKENS", "900")
    )

    configured_temperature = float(
        os.getenv("HF_TEMPERATURE", "0.1")
    )

    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        provider=provider,
        task="text-generation",
        max_new_tokens=(
            max_new_tokens
            if max_new_tokens is not None
            else configured_max_tokens
        ),
        temperature=(
            temperature
            if temperature is not None
            else configured_temperature
        ),
        huggingfacehub_api_token=hf_token,
    )

    return ChatHuggingFace(llm=endpoint)