"""Public exports for the agent helpers used in notebooks."""

from importlib import import_module

__all__ = [
    "Agent",
    "llm",
    "embedding",
    "get_id",
    "tools",
]


def __getattr__(name: str):
    if name == "Agent":
        return import_module(".agent", __name__).Agent
    if name in {"llm", "embedding"}:
        module = import_module(".llms", __name__)
        return getattr(module, name)
    if name == "get_id":
        return import_module(".utils", __name__).get_id
    if name == "tools":
        return import_module(".tools", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
