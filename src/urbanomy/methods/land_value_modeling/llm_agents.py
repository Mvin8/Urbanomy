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
import re
from geopandas import GeoDataFrame
from pathlib import Path
import pdfplumber
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
    """Базовый агент с доступом к RAG для контекста города."""
    
    def __init__(self, name: str, llm: OllamaLLM, rag: CityRAGStore):
        self.name = name
        self.llm = llm
        self.rag = rag
        self._context_cache = {}
    
    def _get_context(self, query: str, k: int = 5) -> list[str]:
        """Получить контекст из RAG с кэшированием."""
        if query not in self._context_cache:
            self._context_cache[query] = self.rag.query(query, k=k)
        return self._context_cache[query]
    
    @abstractmethod
    def evaluate(self, context: dict) -> Dict:
        pass


class CityAdministrationAgent(BaseAgent):
    """Агент, представляющий интересы городской администрации."""
    
    def evaluate(self, context: Dict) -> Dict:
       
        city_context = self._get_context(
            "жилая застройка, плотность, зелёные зоны, инфраструктура",
            k=3
        )
        
        prompt = f"""
        Ты представляешь городскую администрацию. Твоя задача — оценить предложенные изменения с точки зрения развития города, устойчивого развития, соответствия генплану и общественных интересов.

        КОНТЕКСТ ГОРОДА (примеры других районов):
        {chr(10).join(city_context)}

        ПАРАМЕТРЫ РАЙОНА ДЛЯ ОЦЕНКИ:
        {context}

        ИНСТРУКЦИИ:
        - "footprint_area", "build_floor_area", "living_area" даны в кв.м
        - Остальные параметры (residential, business и т.д.) — это доли, сумма = 1
        - Оцени соответствие генплану, баланс функций, влияние на инфраструктуру
        - Укажи, понравилась ли тебе эта схема развития района
        - Ответь кратко и конкретно (2-3 предложения)
        """
        response = self.llm.generate(prompt)
        return response


class ResidentsAgent(BaseAgent):
    """Агент, представляющий интересы жителей."""
    
    def evaluate(self, context: Dict) -> Dict:
        city_context = self._get_context(
            "жилая застройка, население, социальное обеспечение, рекреация",
            k=3
        )
        
        prompt = f"""
        Ты представляешь жителей города. Твоя задача — оценить предложенные изменения с точки зрения качества жизни, комфорта, наличия зелёных зон, социальной инфраструктуры.

        КОНТЕКСТ ГОРОДА (примеры других районов):
        {chr(10).join(city_context)}

        ПАРАМЕТРЫ РАЙОНА ДЛЯ ОЦЕНКИ:
        {context}

        ИНСТРУКЦИИ:
        - "footprint_area", "build_floor_area", "living_area" даны в кв.м
        - Остальные параметры — доли от 0 до 1
        - Оцени плотность, наличие зелёных зон (recreation), социальной инфраструктуры
        - Понравилось ли бы тебе жить в таком районе? Почему?
        - Ответь кратко (2-3 предложения)
        """
        response = self.llm.generate(prompt)
        return response


class InvestorsAgent(BaseAgent):
    """Агент, представляющий интересы инвесторов."""
    
    def evaluate(self, context: Dict) -> Dict:
        city_context = self._get_context(
            "коммерческая застройка, деловые функции, транспорт, доступность",
            k=3
        )
        
        prompt = f"""
        Ты представляешь инвесторов. Твоя задача — оценить предложенные изменения с точки зрения инвестиционной привлекательности, доходности, риска.

        КОНТЕКСТ ГОРОДА (примеры других районов):
        {chr(10).join(city_context)}

        ПАРАМЕТРЫ РАЙОНА ДЛЯ ОЦЕНКИ:
        {context}

        ИНСТРУКЦИИ:
        - "footprint_area", "build_floor_area", "living_area" даны в кв.м
        - Остальные параметры — доли от 0 до 1
        - Оцени баланс жилой и коммерческой застройки, транспортную доступность
        - Видишь ли ты инвестиционный потенциал? Какие риски?
        - Ответь кратко (2-3 предложения)
        """
        response = self.llm.generate(prompt)
        return response

    


class DeveloperAggregator:
    """Агрегатор для обработки предложений от разных заинтересованных сторон."""
    
    def __init__(self, llm: OllamaLLM, agents: list[BaseAgent]):
        self.llm = llm
        self.agents = agents
        self._history = []

    def step(self, genome: Dict) -> Dict:
        """Выполнить один шаг: получить предложения и найти компромисс."""
        proposals = {}
        
      
        for agent in self.agents:
            print(f"[{agent.name}] Evaluating...")
            proposal = agent.evaluate(genome)
            proposals[agent.name] = proposal
        
    
        result = self._aggregate(genome, proposals)
        
       
        self._history.append({
            "genome": genome.copy(),
            "proposals": proposals,
            "result": result
        })
        
        return result

    def _aggregate(self, genome: Dict, proposals: Dict) -> Dict:
        """Найти компромисс между предложениями разных сторон."""
        proposals_text = "\n".join([
            f"- {name}: {proposal}"
            for name, proposal in proposals.items()
        ])
        
        prompt = f"""
        Ты — застройщик и медиатор между разными заинтересованными сторонами: администрацией, жителями и инвесторами.

        ИСХОДНЫЕ ПАРАМЕТРЫ РАЙОНА:
        {self._format_genome(genome)}

        ОЦЕНКИ И ПРЕДЛОЖЕНИЯ СТОРОН:
        {proposals_text}

        ТВОЯ ЗАДАЧА:
        1. Проанализировать предложения всех сторон
        2. Найти сбалансированный компромисс
        3. Убедиться, что параметры физически корректны:
        - Все доли (residential, business и т.д.) должны быть в диапазоне [0, 1]
        - Сумма долей должна быть <= 1
        - footprint_area <= site_area
        - build_floor_area >= footprint_area
        4. Вернуть ОБНОВЛЁННЫЕ параметры в JSON формате

        ОТВЕТ: Только валидный JSON, ничего больше. Пример структуры:
        {json.dumps(genome, ensure_ascii=False, indent=2)}
        """
        
        response = self.llm.generate(prompt)
        print(f"[Aggregator] Generated: {response[:100]}...")
        
        return self._parse(response, genome)

    def _parse(self, text: str, fallback: Dict) -> Dict:
        """Парсить JSON из ответа LLM с валидацией."""
        try:
            
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = text[json_start:json_end]
                parsed = json.loads(json_str)
                
                
                if self._validate_genome(parsed):
                    return parsed
        except Exception as e:
            print(f"[Aggregator] Parse error: {e}")
        
        return fallback.copy()

    def _validate_genome(self, genome: Dict) -> bool:
        """Валидировать генотип на физические ограничения."""
       
        land_use_fields = ["residential", "business", "recreation", 
                          "industrial", "transport", "special", "agriculture"]
        
        total_share = 0
        for field in land_use_fields:
            val = genome.get(field, 0)
            if val < 0 or val > 1:
                return False
            total_share += val
        
        if total_share > 1.0001:  
            return False
        
        
        footprint = genome.get("footprint_area", 0)
        site = genome.get("site_area", 0)
        build = genome.get("build_floor_area", 0)
        
        if footprint > site or build < footprint:
            return False
        
        return True

    def _format_genome(self, genome: Dict) -> str:
        """Форматировать генотип для читаемости."""
        formatted = {}
        for key, value in genome.items():
            if isinstance(value, float):
                formatted[key] = round(value, 3)
            else:
                formatted[key] = value
        
        return json.dumps(formatted, ensure_ascii=False, indent=2)

    def get_history(self) -> list:
        """Получить историю изменений."""
        return self._history.copy()
    
def run_multiagent_loop(
    initial_genome: Dict,
    aggregator: DeveloperAggregator,
    n_steps: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Запустить мультиагентный цикл оптимизации параметров района.
    
    Parameters
    ----------
    initial_genome : Dict
        Исходные параметры района
    aggregator : DeveloperAggregator
        Агрегатор для обработки предложений
    n_steps : int
        Количество итераций
    verbose : bool
        Выводить ли логи процесса
    
    Returns
    -------
    Dict
        Оптимизированные параметры района
    """
    genome = initial_genome.copy()
    
    if verbose:
        print(f"{'='*60}")
        print(f"STARTING MULTIAGENT OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Initial parameters: {json.dumps(initial_genome, ensure_ascii=False, indent=2)}")
        print(f"{'='*60}\n")
    
    for step in range(n_steps):
        if verbose:
            print(f"\n{'='*60}")
            print(f"STEP {step + 1}/{n_steps}")
            print(f"{'='*60}")
        
        prev_genome = genome.copy()
        genome = aggregator.step(genome)
        
        if verbose:
            # Показать изменения
            changes = _compute_genome_delta(prev_genome, genome)
            if changes:
                print(f"\nChanges:")
                for key, (old, new) in changes.items():
                    print(f"  {key}: {old:.3f} → {new:.3f}")
            else:
                print("\nNo changes in this step")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Final parameters: {json.dumps(genome, ensure_ascii=False, indent=2)}")
        print(f"{'='*60}\n")
    
    return genome


def _compute_genome_delta(old: Dict, new: Dict) -> Dict[str, tuple]:
    """Вычислить изменения в параметрах."""
    delta = {}
    
    for key in old.keys():
        if key in new:
            old_val = old[key]
            new_val = new[key]
            
            if isinstance(old_val, (int, float)):
                if abs(old_val - new_val) > 1e-6:
                    delta[key] = (old_val, new_val)
    
    return delta

def load_pdfs(folder):
    pdf_texts = []
    for file in Path(folder).glob("*.pdf"):
        with pdfplumber.open(file) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
        pdf_texts.append({
            "filename": file.name,
            "text": full_text
        })
    return pdf_texts



def clean_text(text: str) -> str:
    """Очистить текст от шумов и лишних пробелов."""
    text = re.sub(r'\n(?=[а-яa-z])', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class PolicyRAGStore:
    """Улучшенное хранилище RAG с гибридным поиском и кэшированием."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.texts = []
        self.embeddings = []
        self.metadata = []
        self._query_cache = {}  
        self._bm25_index = None  
        try:
            from rank_bm25 import BM25Okapi
            self.BM25Okapi = BM25Okapi
            self.use_bm25 = True
        except ImportError:
            self.use_bm25 = False

    def add_documents(self, docs):
        """Добавить документы с семантической чанкизацией."""
        for doc in docs:
            full_text = clean_text(doc["text"])
            chunks = self._smart_chunk_text(full_text)
            
            for chunk_idx, chunk in enumerate(chunks):
                emb = self.embedding_model.encode(chunk)
                
                self.texts.append(chunk)
                self.embeddings.append(emb)
                self.metadata.append({
                    "source": doc["filename"],
                    "chunk_idx": chunk_idx,
                    "chunk_count": len(chunks),
                    "length": len(chunk)
                })

    def _smart_chunk_text(self, text: str, chunk_size=800, overlap=200) -> list:
        """Умная чанкизация с учётом границ предложений."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                words = current_chunk.split()
                overlap_words = words[-overlap//20:] if overlap > 0 else []
                current_chunk = " ".join(overlap_words) + " " + sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def search(self, query: str, top_k: int = 5, use_hybrid: bool = True) -> list:
        """Гибридный поиск с BM25 и embedding-based методами."""
        
        cache_key = (query, top_k)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        if use_hybrid and self.use_bm25 and self.texts:
            results = self._hybrid_search(query, top_k)
        else:
            results = self._embedding_search(query, top_k)
        
        
        if len(self._query_cache) >= 100:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[cache_key] = results
        
        return results

    def _embedding_search(self, query: str, top_k: int) -> list:
        """Поиск только по embedding."""
        if not self.embeddings:
            return []
        
        query_emb = self.embedding_model.encode(query)
        
        sims = [
            (i, query_emb @ emb)
            for i, emb in enumerate(self.embeddings)
        ]
        
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        
        results = []
        for i, score in sims[:top_k]:
            results.append({
                "text": self.texts[i],
                "meta": self.metadata[i],
                "score": float(score)
            })
        
        return results

    def _hybrid_search(self, query: str, top_k: int) -> list:
        """Гибридный поиск: BM25 + embedding с взвешиванием."""
        
        if not self._bm25_index and self.texts:
            tokenized_texts = [text.split() for text in self.texts]
            self._bm25_index = self.BM25Okapi(tokenized_texts)
        
        bm25_scores = {}
        if self._bm25_index:
            query_tokens = query.split()
            bm25_result = self._bm25_index.get_scores(query_tokens)
            bm25_scores = {i: score for i, score in enumerate(bm25_result)}
        
       
        query_emb = self.embedding_model.encode(query)
        embedding_scores = {}
        for i, emb in enumerate(self.embeddings):
            embedding_scores[i] = query_emb @ emb
        
        
        combined_scores = []
        for i in range(len(self.texts)):
            bm25_score = bm25_scores.get(i, 0) / (max(bm25_scores.values()) + 1e-6)
            emb_score = embedding_scores.get(i, 0) / (max(embedding_scores.values()) + 1e-6)
            combined = 0.3 * bm25_score + 0.7 * emb_score
            combined_scores.append((i, combined))
        
        combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
        
        results = []
        for i, score in combined_scores[:top_k]:
            results.append({
                "text": self.texts[i],
                "meta": self.metadata[i],
                "score": float(score)
            })
        
        return results

    def clear_cache(self):
        """Очистить кэш запросов."""
        self._query_cache.clear()
        

class CityRAGBuilder:
    """Построитель документов города с расширенными метаданными."""
    
    def __init__(self, blocks: GeoDataFrame):
        self.blocks = blocks

    def build_docs(self) -> list:
        """Создать документы с богатыми метаданными."""
        docs = []

        for idx, row in self.blocks.iterrows():
            text = self._row_to_text(idx, row)
            metadata = self._extract_metadata(idx, row)

            docs.append({
                "text": text,
                "metadata": metadata
            })

        return docs

    def _extract_metadata(self, idx, row) -> dict:
        """Извлечь расширенные метаданные из строки."""
        fsi = float(row.get("fsi", 0))
        gsi = float(row.get("gsi", 0))
        population = float(row.get("population", 0))
        
        
        if fsi > 3:
            density_class = "high"
        elif fsi > 1:
            density_class = "medium"
        else:
            density_class = "low"
        
        return {
            "district_id": int(idx),
            "land_use": str(row.get("land_use", "unknown")),
            "role": "city_knowledge",
            "fsi": float(fsi),
            "gsi": float(gsi),
            "population": float(population),
            "density_class": density_class,
            "site_area": float(row.get("site_area", 0)),
            "morphotype": str(row.get("morphotype", "unknown")),
            "footprint_area": float(row.get("footprint_area", 0)),
        }

    def _row_to_text(self, idx, row) -> str:
        """Преобразовать строку в текстовое описание."""
        fsi = float(row.get("fsi", 0))
        gsi = float(row.get("gsi", 0))
        population = int(row.get("population", 0))
        land_use = str(row.get("land_use", "неизвестно"))
        morphotype = str(row.get("morphotype", "неизвестно"))
        
        density_desc = (
            'очень высокой' if fsi > 3 else
            'высокой' if fsi > 2 else
            'умеренной' if fsi > 1 else
            'низкой'
        )
        
        return f"""
        Район {idx}

        Основные характеристики:
        - Тип землепользования: {land_use}
        - Морфотип: {morphotype}
        - Площадь участка: {row.get("site_area"):.0f} кв.м
        - Площадь застройки: {row.get("footprint_area"):.0f} кв.м
        - Коэффициент участия: {gsi:.3f}
        - Коэффициент плотности: {fsi:.3f}
        - Население: {population} человек

        Краткая характеристика:
        Район с {density_desc} плотностью застройки (FSI={fsi:.2f}).
        Население составляет {population} человек.
        Морфотип: {morphotype}.
        """
    

class CityRAGStore:
    """Оптимизированное хранилище RAG для контекста города с фильтрацией и ранжированием."""
    
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
        self._query_cache = {}  

    def add_documents(self, docs: list[dict]):
        """Добавить документы с метаданными."""
        texts = [d['text'] for d in docs]
        embeddings = self.embedder.encode(texts).tolist()
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=[d['metadata'] for d in docs],
            ids=[f"doc_{i}" for i in range(len(docs))]
        )

    def query(
        self,
        text: str,
        k: int = 3,
        filters: dict = None,
        rerank: bool = True
    ) -> list[str]:
        """
        Расширенный поиск с фильтрацией и переранжированием.
        
        Parameters
        ----------
        text : str
            Текст запроса
        k : int
            Количество результатов
        filters : dict
            Фильтры по метаданным (например, {"land_use": "residential"})
        rerank : bool
            Использовать переранжирование результатов
        
        Returns
        -------
        list[str]
            Список релевантных текстов
        """
        cache_key = (text, k, tuple(sorted(filters.items())) if filters else None)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        emb = self.embedder.encode([text]).tolist()
        
        
        where_filter = filters if filters else None
        result = self.collection.query(
            query_embeddings=emb,
            n_results=min(k * 3, 20),  
            where=where_filter
        )
        
        documents = result['documents'][0] if result['documents'] else []
        metadatas = result['metadatas'][0] if result['metadatas'] else []
        
       
        if rerank and documents:
            documents, metadatas = self._rerank_results(
                text, documents, metadatas, k
            )
        else:
            documents = documents[:k]
        
        
        if len(self._query_cache) >= 50:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[cache_key] = documents
        
        return documents

    def _rerank_results(
        self,
        query: str,
        documents: list[str],
        metadatas: list[dict],
        k: int
    ) -> tuple[list[str], list[dict]]:
        """Переранжировать результаты по релевантности."""
        scores = []
        
        for doc, meta in zip(documents, metadatas):
            
            query_emb = self.embedder.encode(query)
            doc_emb = self.embedder.encode(doc)
            similarity = query_emb @ doc_emb
            
            
            boost = 1.0
            
            query_words = set(query.lower().split())
            doc_words = set(doc.lower().split())
            word_overlap = len(query_words & doc_words) / (len(query_words) + 1e-6)
            boost += word_overlap * 0.3
            
          
            doc_len = len(doc.split())
            if 50 < doc_len < 300:
                boost += 0.2
            
            scores.append((doc, meta, similarity * boost))
        
       
        scores.sort(key=lambda x: x[2], reverse=True)
        
        reranked_docs = [item[0] for item in scores[:k]]
        reranked_metas = [item[1] for item in scores[:k]]
        
        return reranked_docs, reranked_metas

    def query_by_land_use(
        self,
        land_use: str,
        k: int = 5
    ) -> list[str]:
        """Получить примеры районов с конкретным типом землепользования."""
        return self.query(
            text="",
            k=k,
            filters={"land_use": land_use}
        )

    def query_by_density(
        self,
        density_class: str,
        k: int = 5
    ) -> list[str]:
        """Получить примеры районов с конкретной плотностью."""
        return self.query(
            text="",
            k=k,
            filters={"density_class": density_class}
        )

    def get_similar_districts(
        self,
        query_metadata: dict,
        k: int = 5
    ) -> list[str]:
        """Получить похожие районы по параметрам."""
        description = self._metadata_to_query(query_metadata)
        return self.query(description, k=k)

    def _metadata_to_query(self, meta: dict) -> str:
        """Преобразовать метаданные в текстовый запрос."""
        parts = []
        if "land_use" in meta:
            parts.append(f"Тип землепользования: {meta['land_use']}")
        if "density_class" in meta:
            parts.append(f"Плотность: {meta['density_class']}")
        if "fsi" in meta:
            parts.append(f"FSI около {meta['fsi']:.1f}")
        if "population" in meta:
            parts.append(f"Население: {meta['population']:.0f} человек")
        
        return ". ".join(parts)

    def clear_cache(self):
        """Очистить кэш запросов."""
        self._query_cache.clear()