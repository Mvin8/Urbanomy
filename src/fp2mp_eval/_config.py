import os
from dotenv import load_dotenv

load_dotenv()

class Config():
    def __init__(self):
        self.base_url = os.getenv('FP2MP_CHAT_URL')
        self.api_key = os.getenv('FP2MP_API_KEY')

config = Config()

__all__=[
    "config"
]