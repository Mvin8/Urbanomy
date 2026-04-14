import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv('CHAT_MODEL'),
    base_url=os.getenv('CHAT_URL'),
    api_key=os.getenv('CHAT_API_KEY'),
    temperature=os.getenv('CHAT_TEMPERATURE'),
)

embedding = OpenAIEmbeddings(
    model=os.getenv('EMBEDDING_MODEL'),
    base_url=os.getenv('EMBEDDING_URL'),
    api_key=os.getenv('EMBEDDING_API_KEY')
)

__all__ = [
    'llm',
    'embedding',
]