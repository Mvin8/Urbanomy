import os

from dotenv import load_dotenv


load_dotenv()


class Config:
    def __init__(self) -> None:
        self.base_url = os.getenv("FP2MP_CHAT_URL") or os.getenv("CHAT_URL")
        self.api_key = os.getenv("FP2MP_API_KEY") or os.getenv("CHAT_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("FP2MP_MODEL") or os.getenv("CHAT_MODEL") or "openai/gpt-4.1"


config = Config()

__all__ = ["config"]
