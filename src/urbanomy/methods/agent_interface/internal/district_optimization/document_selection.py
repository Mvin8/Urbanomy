"""RAG helpers for selecting Pareto solutions against a planning document."""

from __future__ import annotations

from typing import Any

from ...models import (
    RetrievedDocumentChunk,
    ParetoDocumentCandidateReview,
    ParetoDocumentSelectionResult,
    StrategyDecisionDocument,
)
from ...tools.internal.district_optimization_formatting import json_value
from ...tools.internal.district_optimization_session import latest_session_or_error
from .document_store import DecisionDocumentStore

_LAND_USE_KEYS = (
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
)


def register_decision_document(
    *,
    llm: Any,
    session_store: dict[str, Any],
    document_name: str | None,
    document_text: str | None,
    document_path: str | None,
    document_type: str | None,
) -> StrategyDecisionDocument:
    """Create and persist the active planning document for one session."""
    source = "inline_text"
    source_ref: str | None = None
    raw_text = str(document_text or "").strip()
    if document_path:
        source = "file_path"
        source_ref = str(document_path).strip()
        inferred_name = source_ref.split("/")[-1].rsplit(".", 1)[0].replace("_", " ").strip()
    else:
        inferred_name = ""
    if not raw_text:
        if not document_path:
            raise ValueError(
                "Не удалось получить текст документа. Передайте document_text или поддерживаемый document_path."
            )
        raw_text = ""
    resolved_name = str(document_name or inferred_name or _infer_document_name(raw_text)).strip()
    resolved_type = _normalize_document_type(document_type=document_type, text=raw_text, name=resolved_name)
    store_factory = session_store.get("decision_document_store_factory", DecisionDocumentStore)
    store = store_factory() if callable(store_factory) else DecisionDocumentStore()
    chunks = store.add_document(
        path=document_path,
        document_name=resolved_name,
        document_text=document_text,
        document_type=resolved_type,
    )
    normalized_text = raw_text or "\n\n".join(chunk.text for chunk in chunks)
    summary, priorities = _extract_document_profile(llm=llm, text=normalized_text)
    document = StrategyDecisionDocument(
        document_name=resolved_name,
        document_type=resolved_type,
        source=source,
        source_ref=source_ref,
        text=normalized_text,
        summary=summary,
        extracted_priorities=priorities,
        chunks=chunks,
        retrieval_backend="ollama_inmemory_vectorstore",
    )
    session_store["active_decision_document"] = document
    session_store["active_decision_document_store"] = store
    return document


def get_active_decision_document(session_store: dict[str, Any]) -> StrategyDecisionDocument:
    """Return the currently active planning document."""
    document = session_store.get("active_decision_document")
    if document is None:
        raise ValueError(
            "Активный документ не найден. Сначала зарегистрируйте документ генплана или стратегии."
        )
    if isinstance(document, StrategyDecisionDocument):
        return document
    return StrategyDecisionDocument.model_validate(document)


def retrieve_document_context(
    *,
    session_store: dict[str, Any],
    query: str,
    top_k: int = 4,
) -> list[RetrievedDocumentChunk]:
    """Retrieve relevant chunks from the active document store."""
    get_active_decision_document(session_store)
    store = session_store.get("active_decision_document_store")
    if store is None:
        raise ValueError(
            "RAG store для активного документа не найден. Сначала заново зарегистрируйте документ."
        )
    return store.retrieve(query=str(query).strip(), top_k=int(top_k))


def select_solution_by_document(
    *,
    llm: Any,
    session_store: dict[str, Any],
    selection_goal: str,
) -> ParetoDocumentSelectionResult:
    """Choose the best Pareto solution against the active planning document."""
    session = latest_session_or_error(session_store)
    document = get_active_decision_document(session_store)
    candidates = _collect_solution_candidates(session)
    if not candidates:
        raise ValueError("В активной optimization session нет Pareto-решений для отбора.")
    retrieved_chunks = retrieve_document_context(
        session_store=session_store,
        query=selection_goal,
        top_k=4,
    )
    scored, criteria_used = _score_candidates(
        candidates=candidates,
        document=document,
        selection_goal=selection_goal,
        retrieved_chunks=retrieved_chunks,
    )
    ranked = sorted(scored, key=lambda item: item["alignment_score"], reverse=True)
    winner = ranked[0]
    candidate_reviews = [
        ParetoDocumentCandidateReview(
            solution_number=int(item["solution_number"]),
            scenario_id=str(item["scenario_id"]),
            title=str(item.get("title", "")).strip(),
            alignment_score=float(item["alignment_score"]),
            alignment_summary=str(item["alignment_summary"]).strip(),
            decisive_factors=[str(value).strip() for value in item.get("decisive_factors", []) if str(value).strip()],
        )
        for item in ranked[: min(4, len(ranked))]
    ]
    rationale = _build_selection_rationale(
        winner=winner,
        document=document,
        criteria_used=criteria_used,
        retrieved_chunks=retrieved_chunks,
    )
    risks = _build_selection_risks(winner=winner, runner_up=ranked[1] if len(ranked) > 1 else None)
    result = ParetoDocumentSelectionResult(
        document_name=document.document_name,
        document_type=document.document_type,
        selection_goal=str(selection_goal).strip() or "Выбрать лучшее решение по документу.",
        document_summary=document.summary,
        selected_solution_number=int(winner["solution_number"]),
        selected_scenario_id=str(winner["scenario_id"]),
        selected_title=str(winner.get("title", "")).strip(),
        rationale=rationale,
        criteria_used=criteria_used,
        retrieved_chunks=retrieved_chunks,
        candidate_reviews=candidate_reviews,
        risks=risks,
    )
    session_store["latest_document_selection_result"] = result
    return result


def latest_document_selection_result(session_store: dict[str, Any]) -> ParetoDocumentSelectionResult:
    """Return the latest persisted document-based selection result."""
    result = session_store.get("latest_document_selection_result")
    if result is None:
        raise ValueError("Пока нет сохранённого выбора решения по документу.")
    if isinstance(result, ParetoDocumentSelectionResult):
        return result
    return ParetoDocumentSelectionResult.model_validate(result)


def _normalize_document_type(*, document_type: str | None, text: str, name: str) -> str:
    value = str(document_type or "").strip().lower()
    if value in {"master_plan", "development_strategy", "custom"}:
        return value
    haystack = f"{name}\n{text}".lower().replace("ё", "е")
    if "генплан" in haystack or "генеральный план" in haystack or "master plan" in haystack:
        return "master_plan"
    if "стратег" in haystack or "strategy" in haystack:
        return "development_strategy"
    return "custom"


def _infer_document_name(text: str) -> str:
    first_line = next((line.strip() for line in str(text).splitlines() if line.strip()), "")
    if first_line:
        return first_line[:80]
    return "Стратегический документ"


def _extract_document_profile(*, llm: Any, text: str) -> tuple[str, list[str]]:
    summary = _fallback_summary(text)
    priorities = _fallback_priorities(text)
    return summary, priorities


def _fallback_summary(text: str) -> str:
    clean = " ".join(str(text).split()).strip()
    if len(clean) <= 240:
        return clean
    return clean[:239].rstrip() + "…"


def _fallback_priorities(text: str) -> list[str]:
    normalized = str(text).lower().replace("ё", "е")
    priorities: list[str] = []
    if any(marker in normalized for marker in ("жиль", "residential", "демограф", "населен")):
        priorities.append("приоритет жилой функции и качества среды")
    if any(marker in normalized for marker in ("делов", "коммер", "бизнес", "рабоч")):
        priorities.append("приоритет деловой активности и рабочих мест")
    if any(marker in normalized for marker in ("парк", "сквер", "озелен", "рекреац", "обществен")):
        priorities.append("приоритет рекреации и общественных пространств")
    if any(marker in normalized for marker in ("инвест", "npv", "окуп", "доход", "эффектив")):
        priorities.append("требование инвестиционной устойчивости")
    if any(marker in normalized for marker in ("стоимост", "земл", "бюджет", "налог")):
        priorities.append("рост стоимости территории и бюджетного эффекта")
    if any(marker in normalized for marker in ("эколог", "выброс", "шум", "зелен")):
        priorities.append("экологические ограничения и снижение негативных воздействий")
    if any(marker in normalized for marker in ("транспорт", "доступн", "мобильн")):
        priorities.append("транспортная доступность")
    if any(marker in normalized for marker in ("промыш", "industrial")):
        priorities.append("контроль промышленной функции")
    return priorities or [
        "сбалансированное развитие территории",
        "учёт как экономического эффекта, так и качества городской среды",
    ]


def _collect_solution_candidates(session: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    frontier = session.pareto_front_df.reset_index(drop=True)
    for index, row in frontier.iterrows():
        payload = {str(key): json_value(value) for key, value in row.items()}
        payload["solution_number"] = int(index) + 1
        payload["scenario_id"] = payload.get("scenario_id") or f"scenario_{index}"
        payload["title"] = str(payload.get("title") or payload["scenario_id"]).strip()
        payload["summary"] = str(payload.get("summary") or "").strip()
        payload["params_repaired"] = dict(payload.get("params_repaired") or {})
        candidates.append(payload)
    return candidates


def _score_candidates(
    *,
    candidates: list[dict[str, Any]],
    document: StrategyDecisionDocument,
    selection_goal: str,
    retrieved_chunks: list[RetrievedDocumentChunk],
) -> tuple[list[dict[str, Any]], list[str]]:
    combined_context = " ".join(
        [
            document.document_name,
            document.summary,
            *document.extracted_priorities,
            *[item.text for item in retrieved_chunks],
            str(selection_goal).strip(),
        ]
    ).lower().replace("ё", "е")
    prefer_high, prefer_low, criteria_used = _preference_profile(combined_context)
    if not criteria_used:
        criteria_used = [
            "сбалансированный рост land_value_gain",
            "достаточная инвестиционная устойчивость",
            "предпочтение смешанным и общественно полезным функциям",
        ]
    metric_ranges = _metric_ranges(candidates)
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        metric_scores: dict[str, float] = {}
        decisive_factors: list[str] = []
        for metric, weight in prefer_high.items():
            metric_score = _candidate_metric_score(
                candidate=candidate,
                metric=metric,
                metric_ranges=metric_ranges,
                prefer_low=False,
            )
            metric_scores[metric] = metric_score
            if weight >= 0.9:
                decisive_factors.append(_describe_metric(metric=metric, score=metric_score, high=True))
        for metric, weight in prefer_low.items():
            metric_score = _candidate_metric_score(
                candidate=candidate,
                metric=metric,
                metric_ranges=metric_ranges,
                prefer_low=True,
            )
            metric_scores[metric] = metric_score
            if weight >= 0.9:
                decisive_factors.append(_describe_metric(metric=metric, score=metric_score, high=False))
        weighted_sum = sum(metric_scores.get(metric, 0.0) * weight for metric, weight in prefer_high.items())
        weighted_sum += sum(metric_scores.get(metric, 0.0) * weight for metric, weight in prefer_low.items())
        total_weight = sum(prefer_high.values()) + sum(prefer_low.values())
        alignment_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        summary = _build_alignment_summary(
            candidate=candidate,
            score=alignment_score,
            decisive_factors=decisive_factors,
            retrieved_chunks=retrieved_chunks,
        )
        scored.append(
            {
                **candidate,
                "alignment_score": round(float(alignment_score), 6),
                "alignment_summary": summary,
                "decisive_factors": list(dict.fromkeys(decisive_factors))[:4],
            }
        )
    return scored, criteria_used


def _metric_ranges(candidates: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for metric in ("land_value_gain", "investor_npv", *_LAND_USE_KEYS):
        values: list[float] = []
        for candidate in candidates:
            value = _raw_metric_value(candidate, metric)
            if value is not None:
                values.append(value)
        if values:
            ranges[metric] = (min(values), max(values))
    return ranges


def _raw_metric_value(candidate: dict[str, Any], metric: str) -> float | None:
    if metric in _LAND_USE_KEYS:
        params = dict(candidate.get("params_repaired") or {})
        value = params.get(metric)
    else:
        value = candidate.get(metric)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_metric_score(
    *,
    candidate: dict[str, Any],
    metric: str,
    metric_ranges: dict[str, tuple[float, float]],
    prefer_low: bool,
) -> float:
    value = _raw_metric_value(candidate, metric)
    if value is None:
        return 0.0
    if metric in _LAND_USE_KEYS:
        normalized = max(0.0, min(1.0, float(value)))
    else:
        min_value, max_value = metric_ranges.get(metric, (value, value))
        span = max_value - min_value
        normalized = 1.0 if span <= 0 else (float(value) - min_value) / span
        normalized = max(0.0, min(1.0, normalized))
    return 1.0 - normalized if prefer_low else normalized


def _preference_profile(text: str) -> tuple[dict[str, float], dict[str, float], list[str]]:
    prefer_high: dict[str, float] = {}
    prefer_low: dict[str, float] = {}
    criteria: list[str] = []

    def _set_high(metric: str, weight: float, description: str) -> None:
        prefer_high[metric] = max(weight, prefer_high.get(metric, 0.0))
        if description not in criteria:
            criteria.append(description)

    def _set_low(metric: str, weight: float, description: str) -> None:
        prefer_low[metric] = max(weight, prefer_low.get(metric, 0.0))
        if description not in criteria:
            criteria.append(description)

    if any(marker in text for marker in ("жиль", "residential", "доступное жилье", "населен")):
        _set_high("residential", 1.0, "повышенная доля жилой функции")
    if any(marker in text for marker in ("делов", "коммер", "бизнес", "рабоч")):
        _set_high("business", 0.85, "деловая активность и рабочие места")
    if any(marker in text for marker in ("парк", "сквер", "озелен", "рекреац", "обществен")):
        _set_high("recreation", 1.0, "рекреация, озеленение и общественные пространства")
    if any(marker in text for marker in ("транспорт", "доступн", "мобильн")):
        _set_high("transport", 0.7, "транспортная доступность")
    if any(marker in text for marker in ("инвест", "npv", "окуп", "доход", "ликвид")):
        _set_high("investor_npv", 1.0, "инвестиционная устойчивость")
    if any(marker in text for marker in ("стоимост", "земл", "налог", "бюджет")):
        _set_high("land_value_gain", 0.95, "рост стоимости территории")
    if any(marker in text for marker in ("эколог", "шум", "выброс", "загряз")):
        _set_high("recreation", 0.8, "экологически более мягкий сценарий")
        _set_low("industrial", 1.0, "снижение промышленной нагрузки")
    if any(marker in text for marker in ("минимизируй промышлен", "меньше промышлен", "без промышлен", "снижение промышлен")):
        _set_low("industrial", 1.0, "минимизация промышленной функции")
    elif any(marker in text for marker in ("промыш", "industrial")):
        _set_high("industrial", 0.6, "сохранение промышленной функции")

    if not criteria:
        _set_high("land_value_gain", 0.7, "рост стоимости территории")
        _set_high("investor_npv", 0.7, "инвестиционная устойчивость")
        _set_high("recreation", 0.35, "более качественная городская среда")
        _set_high("residential", 0.35, "доля жилой функции")
    return prefer_high, prefer_low, criteria


def _describe_metric(*, metric: str, score: float, high: bool) -> str:
    descriptions = {
        "residential": "сильная жилая компонента",
        "business": "заметная деловая компонента",
        "recreation": "сильная рекреационная компонента",
        "industrial": "контролируемая промышленная компонента" if not high else "сильная промышленная компонента",
        "transport": "достаточная транспортная компонента",
        "land_value_gain": "высокий прирост стоимости земли",
        "investor_npv": "высокая инвестиционная устойчивость",
    }
    prefix = descriptions.get(metric, metric)
    if score >= 0.75:
        return prefix
    if score >= 0.45:
        return f"умеренный уровень: {prefix}"
    return f"слабое соответствие по критерию: {metric}"


def _build_alignment_summary(
    *,
    candidate: dict[str, Any],
    score: float,
    decisive_factors: list[str],
    retrieved_chunks: list[RetrievedDocumentChunk],
) -> str:
    base = (
        f"Решение №{candidate['solution_number']} соответствует документу "
        f"на уровне {score:.2f}."
    )
    if decisive_factors:
        base = f"{base} Ключевые основания: {', '.join(decisive_factors[:3])}."
    if retrieved_chunks:
        base = f"{base} Использованы фрагменты {', '.join(item.chunk_id for item in retrieved_chunks[:2])}."
    return base


def _build_selection_rationale(
    *,
    winner: dict[str, Any],
    document: StrategyDecisionDocument,
    criteria_used: list[str],
    retrieved_chunks: list[RetrievedDocumentChunk],
) -> str:
    parts = [
        f"Решение №{winner['solution_number']} лучше всего совпало с приоритетами документа '{document.document_name}'.",
    ]
    if criteria_used:
        parts.append("Ключевые критерии: " + ", ".join(criteria_used[:4]) + ".")
    factors = winner.get("decisive_factors") or []
    if factors:
        parts.append("Сильные стороны сценария: " + ", ".join(factors[:4]) + ".")
    land_use = str((winner.get("params_repaired") or {}).get("land_use") or winner.get("land_use") or "").strip()
    if land_use:
        parts.append(f"Доминирующий land_use: {land_use}.")
    if retrieved_chunks:
        parts.append(
            "RAG-основание: "
            + " | ".join(
                f"{item.chunk_id}: {item.text[:110].strip()}" for item in retrieved_chunks[:2]
            )
            + "."
        )
    return " ".join(parts).strip()


def _build_selection_risks(
    *,
    winner: dict[str, Any],
    runner_up: dict[str, Any] | None,
) -> list[str]:
    risks: list[str] = []
    if runner_up is not None:
        gap = float(winner["alignment_score"]) - float(runner_up["alignment_score"])
        if gap < 0.08:
            risks.append(
                "Отрыв от следующего сценария небольшой, поэтому выбор чувствителен к уточнению критериев документа."
            )
    if float(winner["alignment_score"]) < 0.55:
        risks.append(
            "Ни один сценарий не совпал с документом очень уверенно; возможно, нужны дополнительные ограничения или ручная экспертная проверка."
        )
    return risks
