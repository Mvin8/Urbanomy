'''
LLM agents for land value modeling.
'''
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence
from abc import ABC, abstractmethod
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import random
from geopandas import GeoDataFrame
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy
from blocksnet.analysis.indicators import calculate_density_indicators
from blocksnet.analysis.morphotypes import get_strelka_morphotypes
from blocksnet.enums import LandUse

from .constants import (
    BlockColumn,
    CATEGORICAL_FEATURES,
    ORIGINAL_FEATURES,
    RADIUS_LIST,
    ScenarioResultKey,
)
from .land_price_estimation import LandPriceEstimator

import requests
import json
from typing import Any, Optional

class OllamaLLM:
    def __init__(self,
                 model: str = 'llama3',
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.7):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        
    def generate(self, prompt: str, system: Optional[str] = None ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False
        }
        if system:
            payload["system"] = system
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300
        ) 
        response.raise_for_status()
        result = response.json()
        return result['response']


class BaseAgent(ABC):
    def __init__(slef,name:str, llm: OllamaLLM):
        slef.name = name
        slef.llm = llm
    @abstractmethod
    def evaluate(self, context: dict) -> Dict:
        pass

class CityAdministrationAgent(BaseAgent):
    def evaluate(self, context: Dict) -> Dict:
        prompt = f"""
        Ты представляешь администрацию города.

        Текущие параметры района:
        {context}

        Твоя задача:
        - снизить транспортную нагрузку
        - обеспечить устойчивую плотность
        - избегать перекоса в промышленность

        Что бы ты хотел изменить?
        """

        response = self.llm.generate(prompt)
        print(f"Response from {self.name}: {response}")
        return response


class ResidentsAgent(BaseAgent):
    def evaluate(self, context: Dict) -> Dict:
        prompt = f"""
        Ты представляешь жителей соседних районов.

        Параметры района:
        {context}

        Приоритеты:
        - зелёные зоны
        - низкая плотность
        - комфортная инфраструктура

        Что бы ты хотел изменить?
        """

        response = self.llm.generate(prompt)
        print(f"Response from {self.name}: {response}")
        
        return response
        

class InvestorsAgent(BaseAgent):
    def evaluate(self, context: Dict) -> Dict:
        prompt = f"""
        Ты представляешь инвесторов.

        Параметры района:
        {context}

        Цели:
        - рост стоимости земли
        - эффективность застройки
        - допустим рост business

        Что бы ты хотел изменить?
        """

        response = self.llm.generate(prompt)
        print(f"Response from {self.name}: {response}")
        
        return response

    


class DeveloperAggregator:
    def __init__(self, llm: OllamaLLM, agents: list[BaseAgent]):
        self.llm = llm
        self.agents = agents

    def step(self, genome: Dict) -> Dict:
        proposals = {}

        for agent in self.agents:
            delta = agent.evaluate(genome)
            proposals[agent.name] = delta

        return self._aggregate(genome, proposals)

    def _aggregate(self, genome: Dict, proposals: Dict) -> Dict:
        prompt = f"""
        Ты — застройщик и медиатор.

        Исходные параметры:
        {genome}

        Предложения сторон:
        {proposals}

        Твоя задача:
        - найти компромисс
        - сохранить физические ограничения
        - вернуть ОБНОВЛЁННЫЕ параметры района в JSON
        Верни ТОЛЬКО валидный JSON.
        Без комментариев.
        Без пояснений.
        Без markdown.
        Без текста до или после.
        Строго формата как в текущем параметре района.
        """

        response = self.llm.generate(prompt)
        print(f"Response from aggregator: {response}")
        print("Returned genome:", self._parse(response, genome))
        return self._parse(response, genome)

    def _parse(self, text: str, genome: Dict) -> Dict:
        import json
        try:
            return json.loads(text)
        except Exception:
            return genome.copy()
        
def run_multiagent_loop(
    initial_genome: Dict,
    aggregator: DeveloperAggregator,
    n_steps: int = 5,
):
    genome = initial_genome.copy()

    for step in range(n_steps):
        print(f"\n=== STEP {step} ===")
        genome = aggregator.step(genome)

    return genome