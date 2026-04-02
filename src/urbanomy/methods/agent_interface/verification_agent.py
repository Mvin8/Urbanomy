"""Deterministic verification layer for Urbanomy domain-agent outputs."""

from __future__ import annotations

from typing import Any

from .models import VerificationIssue, VerificationReport


class VerificationAgent:
    """Validate runtime outputs and produce compact verification reports."""

    def verify(
        self,
        *,
        route: str,
        result: Any,
        optimization_summary: dict[str, Any] | None = None,
    ) -> VerificationReport:
        if route == "visualization":
            return self._verify_visualization(result)
        if route == "district_optimization":
            return self._verify_district_optimization(
                result=result,
                optimization_summary=optimization_summary,
            )
        if route == "document_rag":
            return self._verify_document_rag(result)
        if route == "block_parameters":
            return self._verify_block_parameters(result)
        if route == "general_qa":
            return self._verify_general_qa(result)
        return VerificationReport(
            status="warning",
            summary="Для этого маршрута нет специализированной проверки.",
            issues=[VerificationIssue(severity="warning", message="Маршрут не покрыт verifier layer.")],
        )

    @staticmethod
    def _verify_visualization(result: Any) -> VerificationReport:
        issues: list[VerificationIssue] = []
        facts: list[str] = []
        route = str(getattr(result, "route", "")).strip()
        title = str(getattr(result, "title", "")).strip()
        payload = dict(getattr(result, "tool_payload", {}) or {})
        if not route:
            issues.append(VerificationIssue(severity="error", message="У visualization-результата пустой route."))
        else:
            facts.append(f"Выбран route={route}.")
        if title:
            facts.append(f"Построен артефакт с заголовком '{title}'.")
        else:
            issues.append(VerificationIssue(severity="warning", message="У результата визуализации пустой title."))
        if not payload:
            issues.append(
                VerificationIssue(
                    severity="warning",
                    message="У результата визуализации пустой tool_payload, follow-up контекст будет беднее.",
                )
            )
        if route == "plot_target_block_map":
            target_id = getattr(result, "target_id", None)
            if not isinstance(target_id, int):
                issues.append(
                    VerificationIssue(
                        severity="error",
                        message="Для plot_target_block_map не зафиксирован target_id.",
                    )
                )
            else:
                facts.append(f"В карте выделен квартал id={target_id}.")
        else:
            metric_kind = str(getattr(result, "metric_kind", "")).strip()
            price_column = str(getattr(result, "price_column", "")).strip()
            if metric_kind:
                facts.append(f"Использована метрика {metric_kind}.")
            else:
                issues.append(
                    VerificationIssue(severity="warning", message="У visualization-результата пустой metric_kind.")
                )
            if not price_column:
                issues.append(
                    VerificationIssue(severity="warning", message="У visualization-результата пустой price_column.")
                )
        if getattr(result, "artifact", None) is None:
            issues.append(
                VerificationIssue(
                    severity="warning",
                    message="Матplotlib artifact не сохранен в runtime output.",
                )
            )
        return _report_from_issues(
            issues=issues,
            facts=facts,
            ok_summary="Результат визуализации согласован с ожидаемой схемой.",
            warning_summary="Результат визуализации построен, но часть артефактов неполная.",
            error_summary="Визуализация завершилась с нарушением ожидаемой схемы результата.",
        )

    @staticmethod
    def _verify_district_optimization(
        *,
        result: Any,
        optimization_summary: dict[str, Any] | None,
    ) -> VerificationReport:
        issues: list[VerificationIssue] = []
        facts: list[str] = []
        if not isinstance(result, dict):
            return VerificationReport(
                status="error",
                summary="Optimization-результат не является словарем.",
                issues=[VerificationIssue(severity="error", message="Ожидался dict с полями status/response/tool_output.")],
            )
        status = str(result.get("status", "")).strip()
        tool_name = str(result.get("tool_name") or "").strip()
        tool_output = result.get("tool_output")
        if status != "ok":
            issues.append(
                VerificationIssue(
                    severity="error",
                    message=f"Optimization agent вернул status={status or 'unknown'}.",
                )
            )
        else:
            facts.append("Optimization agent вернул status=ok.")
        if tool_name:
            facts.append(f"Использован инструмент {tool_name}.")
        if isinstance(tool_output, dict):
            solution_number = tool_output.get("solution_number")
            if isinstance(solution_number, int):
                facts.append(f"В ответе фигурирует решение №{solution_number}.")
                n_solutions = _safe_int((optimization_summary or {}).get("n_solutions"))
                if n_solutions is not None and not (1 <= solution_number <= n_solutions):
                    issues.append(
                        VerificationIssue(
                            severity="error",
                            message=(
                                f"solution_number={solution_number} выходит за диапазон доступных решений "
                                f"1..{n_solutions}."
                            ),
                        )
                    )
            if tool_name == "run_district_optimization":
                if optimization_summary is None:
                    issues.append(
                        VerificationIssue(
                            severity="error",
                            message="После run_district_optimization не сформирована summary сессии.",
                        )
                    )
                else:
                    target_id = optimization_summary.get("target_id")
                    n_solutions = optimization_summary.get("n_solutions")
                    facts.append(f"Активная сессия target_id={target_id}, найдено решений={n_solutions}.")
            if tool_name == "select_pareto_solution_by_document":
                document_name = str(tool_output.get("document_name") or "").strip()
                if document_name:
                    facts.append(f"Выбор выполнен по документу '{document_name}'.")
                else:
                    issues.append(
                        VerificationIssue(
                            severity="error",
                            message="Document-based selector вернул пустой document_name.",
                        )
                    )
                rationale = str(tool_output.get("rationale") or "").strip()
                if not rationale:
                    issues.append(
                        VerificationIssue(
                            severity="warning",
                            message="Document-based selector не вернул текстовое обоснование выбора.",
                        )
                    )
                retrieved_chunks = list(tool_output.get("retrieved_chunks") or [])
                if retrieved_chunks:
                    facts.append(f"Для выбора использовано RAG-фрагментов: {len(retrieved_chunks)}.")
                else:
                    issues.append(
                        VerificationIssue(
                            severity="warning",
                            message="Document-based selector не вернул retrieved_chunks с evidence-контекстом.",
                        )
                    )
        elif tool_name:
            issues.append(
                VerificationIssue(
                    severity="warning",
                    message=f"Для инструмента {tool_name} отсутствует dict tool_output.",
                )
            )
        return _report_from_issues(
            issues=issues,
            facts=facts,
            ok_summary="Optimization-ответ согласован с активной session state.",
            warning_summary="Optimization-ответ получен, но часть session metadata неполная.",
            error_summary="Optimization-ответ не прошел согласованность с session state.",
        )

    @staticmethod
    def _verify_document_rag(result: Any) -> VerificationReport:
        issues: list[VerificationIssue] = []
        facts: list[str] = []
        if not isinstance(result, dict):
            return VerificationReport(
                status="error",
                summary="Document-RAG результат не является словарем.",
                issues=[VerificationIssue(severity="error", message="Ожидался dict с полями status/response/tool_output.")],
            )
        status = str(result.get("status", "")).strip()
        tool_name = str(result.get("tool_name") or "").strip()
        tool_output = result.get("tool_output")
        if status != "ok":
            issues.append(
                VerificationIssue(
                    severity="error",
                    message=f"Document-RAG agent вернул status={status or 'unknown'}.",
                )
            )
        else:
            facts.append("Document-RAG agent вернул status=ok.")
        if tool_name:
            facts.append(f"Использован инструмент {tool_name}.")
        if not isinstance(tool_output, dict):
            issues.append(
                VerificationIssue(
                    severity="warning",
                    message="Document-RAG ответ не содержит dict tool_output.",
                )
            )
            return _report_from_issues(
                issues=issues,
                facts=facts,
                ok_summary="Document-RAG ответ согласован с ожидаемой схемой.",
                warning_summary="Document-RAG ответ получен, но metadata неполная.",
                error_summary="Document-RAG ответ не прошел базовую проверку схемы.",
            )
        if tool_name == "register_decision_document":
            document_name = str(tool_output.get("document_name") or "").strip()
            chunk_count = tool_output.get("chunk_count")
            if document_name:
                facts.append(f"Зарегистрирован документ '{document_name}'.")
            else:
                issues.append(
                    VerificationIssue(
                        severity="error",
                        message="Регистрация документа не вернула document_name.",
                    )
                )
            if isinstance(chunk_count, int) and chunk_count > 0:
                facts.append(f"Индексировано чанков: {chunk_count}.")
            else:
                issues.append(
                    VerificationIssue(
                        severity="warning",
                        message="Регистрация документа не вернула положительный chunk_count.",
                    )
                )
        elif tool_name == "get_active_decision_document":
            document_name = str(tool_output.get("document_name") or "").strip()
            if document_name:
                facts.append(f"Активный документ: '{document_name}'.")
            else:
                issues.append(
                    VerificationIssue(
                        severity="warning",
                        message="Запрос active document не вернул document_name.",
                    )
                )
        elif tool_name == "retrieve_decision_document_context":
            chunks = list(tool_output.get("chunks") or [])
            if chunks:
                facts.append(f"Найдено релевантных фрагментов: {len(chunks)}.")
            else:
                issues.append(
                    VerificationIssue(
                        severity="warning",
                        message="RAG retrieval не вернул ни одного чанка.",
                    )
                )
        elif tool_name == "select_pareto_solution_by_document":
            selected_solution_number = tool_output.get("selected_solution_number")
            if isinstance(selected_solution_number, int):
                facts.append(f"Выбрано решение №{selected_solution_number}.")
            else:
                issues.append(
                    VerificationIssue(
                        severity="error",
                        message="Document-based selector не вернул selected_solution_number.",
                    )
                )
            retrieved_chunks = list(tool_output.get("retrieved_chunks") or [])
            if retrieved_chunks:
                facts.append(f"Для выбора использовано RAG-фрагментов: {len(retrieved_chunks)}.")
            else:
                issues.append(
                    VerificationIssue(
                        severity="warning",
                        message="Document-based selector не вернул retrieved_chunks.",
                    )
                )
        return _report_from_issues(
            issues=issues,
            facts=facts,
            ok_summary="Document-RAG ответ согласован с ожидаемой схемой.",
            warning_summary="Document-RAG ответ получен, но metadata неполная.",
            error_summary="Document-RAG ответ не прошел базовую проверку схемы.",
        )

    @staticmethod
    def _verify_block_parameters(result: Any) -> VerificationReport:
        issues: list[VerificationIssue] = []
        facts: list[str] = []
        if not isinstance(result, dict):
            return VerificationReport(
                status="error",
                summary="Результат block_parameters не является словарем.",
                issues=[VerificationIssue(severity="error", message="Ожидался dict с полями status/target_id/parameters.")],
            )
        status = str(result.get("status", "")).strip()
        target_id = result.get("target_id")
        parameters = result.get("parameters")
        if status != "ok":
            issues.append(
                VerificationIssue(
                    severity="error",
                    message=f"Block parameters agent вернул status={status or 'unknown'}.",
                )
            )
        if isinstance(target_id, int):
            facts.append(f"Параметры получены для квартала id={target_id}.")
        else:
            issues.append(VerificationIssue(severity="error", message="В block parameters результате нет корректного target_id."))
        if isinstance(parameters, dict) and parameters:
            facts.append(f"Возвращено параметров: {len(parameters)}.")
        else:
            issues.append(VerificationIssue(severity="warning", message="Block parameters вернул пустой набор параметров."))
        return _report_from_issues(
            issues=issues,
            facts=facts,
            ok_summary="Параметры квартала согласованы с ожидаемой схемой результата.",
            warning_summary="Параметры квартала получены, но ответ неполный.",
            error_summary="Ответ block parameters нарушает ожидаемую схему.",
        )

    @staticmethod
    def _verify_general_qa(result: Any) -> VerificationReport:
        text = str(result).strip()
        if not text:
            return VerificationReport(
                status="warning",
                summary="General QA вернул пустой ответ.",
                issues=[VerificationIssue(severity="warning", message="Разговорная ветка не сформировала текст ответа.")],
            )
        return VerificationReport(
            status="ok",
            summary="General QA вернул непустой текстовый ответ.",
            verified_facts=["Разговорная ветка сгенерировала ответ пользователю."],
        )


def create_verification_agent() -> VerificationAgent:
    """Factory for the deterministic verifier layer."""
    return VerificationAgent()


def _report_from_issues(
    *,
    issues: list[VerificationIssue],
    facts: list[str],
    ok_summary: str,
    warning_summary: str,
    error_summary: str,
) -> VerificationReport:
    if any(item.severity == "error" for item in issues):
        status = "error"
        summary = error_summary
    elif issues:
        status = "warning"
        summary = warning_summary
    else:
        status = "ok"
        summary = ok_summary
    return VerificationReport(
        status=status,
        summary=summary,
        issues=issues,
        verified_facts=facts,
    )


def _safe_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None
