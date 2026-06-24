import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

from .config import config

load_dotenv()

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def init_llm(model: str | None = None, temperature: float = 0.5):
    return ChatOpenAI(
        model=model or config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=temperature,
    )


llm = init_llm(temperature=_float_env("CHAT_TEMPERATURE", 0.5))

embedding = OpenAIEmbeddings(
    model=os.getenv('EMBEDDING_MODEL') or 'text-embedding-3-small',
    base_url=os.getenv('EMBEDDING_URL'),
    api_key=os.getenv('EMBEDDING_API_KEY') or config.api_key,
)

__all__ = [
    'config',
    'init_llm',
    'llm',
    'embedding',
]
