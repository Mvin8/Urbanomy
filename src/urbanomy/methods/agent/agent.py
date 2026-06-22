import uuid

from langchain.agents import create_agent
from pydantic import BaseModel

from .llms import llm


class Agent:
    """Tiny notebook helper around langchain.create_agent."""

    def __init__(
        self,
        system_prompt: str,
        *args,
        response_format: type[BaseModel] | None = None,
        tools: list | None = None,
        **kwargs,
    ):
        self.id = uuid.uuid4().hex[:6]
        self._agent = create_agent(
            *args,
            model=llm,
            tools=tools,
            response_format=response_format,
            system_prompt=system_prompt,
            **kwargs,
        )

    def invoke(self, messages):
        response = self._agent.invoke(input={"messages": messages})
        return response.get("structured_response", response["messages"][-1].content)


__all__ = ["Agent"]
