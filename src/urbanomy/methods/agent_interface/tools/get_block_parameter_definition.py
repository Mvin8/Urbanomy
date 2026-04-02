"""Tool for explaining Urbanomy block-parameter terms."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..internal.block_parameters.glossary import (
    BLOCK_PARAMETER_DEFINITIONS,
    build_block_parameter_definition_text,
    detect_block_parameter_term,
)


class GetBlockParameterDefinitionInput(BaseModel):
    """Input schema for the block-parameter glossary tool."""

    model_config = ConfigDict(extra="forbid")

    term: str = Field(description="Parameter term like fsi, gsi, mxi, population, site_area.")

    @field_validator("term", mode="before")
    @classmethod
    def _normalize_term(cls, value: Any) -> str:
        text = str(value).strip().lower().replace("ё", "е")
        detected = detect_block_parameter_term(text)
        return detected or text


def make_get_block_parameter_definition_tool():
    """Create a LangChain tool that explains one block-parameter term."""

    @tool(
        "get_block_parameter_definition",
        args_schema=GetBlockParameterDefinitionInput,
    )
    def get_block_parameter_definition(term: str) -> dict[str, Any]:
        """What the tool does.

        Returns a short glossary explanation for one Urbanomy block parameter
        such as FSI, GSI, MXI, population, site_area or footprint_area.

        When to use this tool.

        Use this tool when the user asks what a block parameter means, how it
        is calculated, or what it is used for.

        Args:
            term: Parameter name or alias.

        Returns:
            dict[str, Any]: Structured glossary payload and a compact text answer.
        """
        normalized_term = detect_block_parameter_term(term) or str(term).strip().lower()
        item = BLOCK_PARAMETER_DEFINITIONS.get(normalized_term)
        if item is None:
            available_terms = ", ".join(sorted(BLOCK_PARAMETER_DEFINITIONS))
            return {
                "term": normalized_term,
                "found": False,
                "available_terms": sorted(BLOCK_PARAMETER_DEFINITIONS),
                "definition_text": (
                    f"Не удалось найти определение для `{normalized_term}`. "
                    f"Доступные термины: {available_terms}."
                ),
            }
        return {
            "term": normalized_term,
            "found": True,
            "label": item["label"],
            "definition": item["definition"],
            "formula": item["formula"],
            "interpretation": item["interpretation"],
            "definition_text": build_block_parameter_definition_text(normalized_term),
        }

    return get_block_parameter_definition
