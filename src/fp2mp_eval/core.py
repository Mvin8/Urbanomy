
import pandas as pd
# from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tqdm.contrib.concurrent import thread_map
from .models import Evaluation, Dimension
from ._prompt import EVAL_PROMPT
from ._config import config

class FP2MPEval():

    def __init__(self, model : str = 'openai/gpt-4.1', temperature : float = 0.5, n_judges : int = 10, **kwargs):
        if model == 'llama3':
            self._llm = ChatOllama(model=model, temperature=temperature
                                  )
        else:
            self._llm = ChatOpenAI(
                model=model,
                base_url=config.base_url,
                api_key=config.api_key,
                temperature=temperature,
                **kwargs
            )
        self._n_judges = n_judges

    @property
    def llm(self) -> ChatOpenAI|ChatOllama:
        return self._llm.with_structured_output(Evaluation)
    
    def evaluate_case(self, case : tuple[str,str], max_workers : int = 2) -> list[Evaluation]:
        problem, solution = case
        message = EVAL_PROMPT.format(problem=problem, solution=solution)
        n_judges = self._n_judges

        def invoke(*args, **kwargs):
            return self.llm.invoke(message)

        results = thread_map(invoke, range(n_judges), max_workers=max_workers)

        return results
    
    @staticmethod
    def evaluations_to_df(evaluations : list[Evaluation]) -> pd.DataFrame:
        data = []

        for evaluation in evaluations:
            d = {}
            dump = evaluation.model_dump()
            for k,v in dump.items():
                d[k] = v['score']
            data.append(d)
            
        return pd.DataFrame(data)
    
    @staticmethod
    def evaluations_to_long_df(evaluations : list[Evaluation]) -> pd.DataFrame:
        data = []

        for judge, evaluation in enumerate(evaluations):
            for indicator, v in evaluation.model_dump().items():
                data.append({
                    'judge': judge,
                    'indicator': indicator,
                    **v
                })

        return pd.DataFrame(data)