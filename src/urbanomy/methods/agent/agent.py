from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import BaseMessage
from langchain.agents import create_agent
from .llms import llm
from .utils import get_id

SUMMARIZATION_KWARGS = {
    'trigger': ('tokens', 30000),
    'keep': ('messages', 15)
}
STRUCTURED_RESPONSE_KEY = 'structured_response'
MESSAGES_KEY = 'messages'

class Agent():

    def __init__(
        self,
        system_prompt : str,
        *args,
        response_format : type[BaseModel] | None = None,
        tools : list | None = None,
        metadata : dict | None = None,
        is_checkpointer : bool = False,
        is_summarization : bool = False,
        summarization_kwargs : dict | None = None,
        **kwargs
    ):
        self.id = get_id()
        
        self._system_prompt = system_prompt
        self.response_format = response_format
        self.tools = tools
        self._metadata = metadata or {}

        if is_checkpointer:
            if response_format is not None: # FIXME CRITICAL
                import langgraph.checkpoint.serde
                langgraph.checkpoint.serde._msgpack.SAFE_MSGPACK_TYPES = (
                    langgraph.checkpoint.serde._msgpack.SAFE_MSGPACK_TYPES.union({(response_format.__module__, response_format.__name__)})
                )
            self.checkpointer = InMemorySaver()
        else:
            self.checkpointer = None

        if is_summarization:
            summarization_kwargs = summarization_kwargs or SUMMARIZATION_KWARGS
            self.middleware = [SummarizationMiddleware(
                model=llm,
                **summarization_kwargs
            )]
        else:
            self.middleware = []
        
        self._agent = create_agent(
            *args,
            model=llm, 
            tools=tools, 
            response_format=response_format, 
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            middleware=self.middleware,
            **kwargs
        )

    def invoke(self, messages):
        response = self._agent.invoke(
            input={'messages': messages},
            config=self.runnable_config if self.checkpointer is not None else None
        )
        if STRUCTURED_RESPONSE_KEY in response:
            return response[STRUCTURED_RESPONSE_KEY]
        return response[MESSAGES_KEY][-1].content
    
    @property
    def metadata(self):
        return {
            'id': self.id,
            **self._metadata
        }
    
    @property
    def system_prompt(self):
        return self._system_prompt
        # return self._system_prompt.format(**self.metadata)
    
    @property
    def runnable_config(self):
        return {"configurable": {"thread_id": self.id}}
    
    @property
    def messages(self) -> list:
        if self.checkpointer is None:
            return []
        thread = self.checkpointer.get(self.runnable_config)
        return thread['channel_values']['messages']
    
__all__=[
    'Agent'
]