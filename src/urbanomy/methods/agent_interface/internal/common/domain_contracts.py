"""Shared contracts for exposing domain-agent capabilities and tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ToolDescriptor:
    """Compact user-facing description of one available tool."""

    name: str
    description: str


def describe_tool(tool: Any) -> ToolDescriptor:
    """Build a compact descriptor from a LangChain tool object."""
    name = str(getattr(tool, "name", "")).strip()
    description = str(getattr(tool, "description", "")).strip().split("\n\n", 1)[0].strip()
    return ToolDescriptor(name=name, description=description)


def format_tool_catalog(descriptors: Iterable[ToolDescriptor]) -> str:
    """Render tool descriptors into a readable catalog block."""
    lines: list[str] = []
    for descriptor in descriptors:
        name = str(descriptor.name).strip()
        if not name:
            continue
        description = str(descriptor.description).strip()
        if description:
            lines.append(f"- {name}: {description}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)
