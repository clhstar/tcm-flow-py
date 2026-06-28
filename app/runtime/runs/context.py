from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.store.models import RunRecord


@dataclass(frozen=True)
class RunContext:
    thread_store: Any
    agent_context: Mapping[str, Any] = field(default_factory=dict)


def build_runtime_context(
    record: RunRecord,
    agent_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    runtime_context = dict(agent_context or {})
    runtime_context["thread_id"] = record.thread_id
    runtime_context["run_id"] = record.run_id
    return runtime_context


def build_runnable_config(
    record: RunRecord,
    request_config: Mapping[str, Any] | None,
    runtime_context: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(request_config or {})

    raw_configurable = config.get("configurable")
    configurable = (
        dict(raw_configurable) if isinstance(raw_configurable, Mapping) else {}
    )
    configurable["thread_id"] = record.thread_id
    config["configurable"] = configurable

    raw_context = config.get("context")
    installed_context = dict(raw_context) if isinstance(raw_context, Mapping) else {}
    installed_context.update(runtime_context)
    config["context"] = installed_context

    config.setdefault(
        "recursion_limit",
        int(runtime_context.get("recursion_limit", 50)),
    )
    return config
