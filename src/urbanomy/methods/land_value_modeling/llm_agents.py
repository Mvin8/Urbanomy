'''
LLM agents for land value modeling.
'''
from __future__ import annotations
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
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
    def __init__(self,name:str, llm: OllamaLLM, rag: CityRAGStore):
        self.name = name
        self.llm = llm
        self.rag = rag
    @abstractmethod
    def evaluate(self, context: dict) -> Dict:
        pass

class CityAdministrationAgent(BaseAgent):
    def evaluate(self, context: Dict) -> Dict:
        city_context = self.rag.query(text="жилая застройка, плотность, зелёные зоны", k=10)
        prompt = f"""
        Ты представляешь городскую администрацию в задаче изменений параметров застройки района.
        Контекст города для ознакомления с тем, какие бывают характеристики застройки районов в городе:
        {chr(10).join(city_context)}

        Параметры района, который нужно поменять:
        {context}
        Тут параметры "footprint_area","build_floor_area","living_area" даны в квадратных метрах, остальные параметры - доли, дающие в сумме единиицу

        Что бы ты хотел изменить?
        Меняй ТОЛЬКО тот район, который нужно изменить, не трогай районы из контекста!
        Описывая изменения старайся выдавать численный результат.
        Опиши это двумя абзацеми.
        """

        response = self.llm.generate(prompt)
        print(f"Response from {self.name}: {response}")
        return response


class ResidentsAgent(BaseAgent):
    def evaluate(self, context: Dict) -> Dict:
        city_context = self.rag.query(text="жилая застройка, плотность, зелёные зоны", k=10)
        prompt = f"""
        Ты представляешь жителей города в задаче изменений параметров застройки района.
        Контекст города для ознакомления с тем, какие бывают характеристики застройки районов в городе:
        {chr(10).join(city_context)}

        Параметры района, который нужно поменять:
        {context}
        Тут параметры "footprint_area","build_floor_area","living_area" даны в квадратных метрах, остальные параметры - доли, дающие в сумме единиицу
        Что бы ты хотел изменить?
        Описывая изменения старайся выдавать численный результат.
        Меняй ТОЛЬКО тот район, который нужно изменить, не трогай районы из контекста!
        Опиши это двумя абзацеми.
        """

        response = self.llm.generate(prompt)
        print(f"Response from {self.name}: {response}")
        
        return response
        

class InvestorsAgent(BaseAgent):
    def evaluate(self, context: Dict) -> Dict:
        city_context = self.rag.query(text="жилая застройка, плотность, зелёные зоны", k=10)
        prompt = f"""
        Ты представляешь инвесторов в задаче изменений параметров застройки района.
        Контекст города для ознакомления с тем, какие бывают характеристики застройки районов в городе:
        {chr(10).join(city_context)}

        Параметры района, который нужно поменять:
        {context}
        Тут параметры "footprint_area","build_floor_area","living_area" даны в квадратных метрах, остальные параметры - доли, дающие в сумме единиицу
        Что бы ты хотел изменить?
        Меняй ТОЛЬКО тот район, который нужно изменить, не трогай районы из контекста!
        Описывая изменения старайся выдавать численный результат.
        Опиши это двумя абзацеми.
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


class CityRAGBuilder:
    def __init__(self, blocks):
        self.blocks = blocks

    def build_docs(self):
        docs = []

        for idx, row in self.blocks.iterrows():
            text = self._row_to_text(idx, row)

            docs.append({
                "text": text,
                "metadata": {
                    "district_id": int(idx),
                    "land_use": str(row.get("land_use")),
                    "role": "city_knowledge"
                }
            })

        return docs

    def _row_to_text(self, idx, row):
        return f"""
        Район {idx}

        Тип землепользования: {row.get("land_use")}
        Площадь пятна застройки: {row.get("footprint_area")}
        Площадь участка: {row.get("site_area")}
        FSI: {row.get("fsi")}
        GSI: {row.get("gsi")}
        Население: {row.get("population")}

        Краткая характеристика:
        Район с {'высокой' if row.get('fsi', 0) > 2 else 'умеренной'} плотностью.
        """
    

class CityRAGStore:
    def __init__(
            self,
            persist_dir: str = "./city_rag",
            collection_name: str = "city",
            
    ):
        self.client = chromadb.Client(
            Settings(persist_directory=persist_dir, anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer("all-MiniLM-l6-v2")

    def add_documents(self, docs:list[dict]):
        texts = [d['text'] for d in docs]
        embeddings = self.embedder.encode(texts).tolist()
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=[d['metadata'] for d in docs],
            ids=[f"doc_{i}" for i in range(len(docs))]
        )

    def query(self, text: str, k: int = 3) -> list[str]:
        emb = self.embedder.encode([text]).tolist()

        result = self.collection.query(
            query_embeddings=emb,
            n_results=k          
        )
        return result['documents'][0]