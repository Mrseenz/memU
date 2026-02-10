from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "BlobConfig",
    "DatabaseConfig",
    "DefaultUserModel",
    "LLMConfig",
    "LLMProfilesConfig",
    "LocalWorkflowRunner",
    "MemorizeConfig",
    "MemoryService",
    "RetrieveConfig",
    "UserConfig",
    "WorkflowRunner",
    "register_workflow_runner",
    "resolve_workflow_runner",
]

if TYPE_CHECKING:
    from memu.app.service import MemoryService
    from memu.app.settings import (
        BlobConfig,
        DatabaseConfig,
        DefaultUserModel,
        LLMConfig,
        LLMProfilesConfig,
        MemorizeConfig,
        RetrieveConfig,
        UserConfig,
    )
    from memu.workflow.runner import (
        LocalWorkflowRunner,
        WorkflowRunner,
        register_workflow_runner,
        resolve_workflow_runner,
    )


def __getattr__(name: str) -> Any:
    if name == "MemoryService":
        from memu.app.service import MemoryService as value

        return value

    if name in {
        "BlobConfig",
        "DatabaseConfig",
        "DefaultUserModel",
        "LLMConfig",
        "LLMProfilesConfig",
        "MemorizeConfig",
        "RetrieveConfig",
        "UserConfig",
    }:
        from memu.app import settings as _settings

        return getattr(_settings, name)

    if name in {
        "LocalWorkflowRunner",
        "WorkflowRunner",
        "register_workflow_runner",
        "resolve_workflow_runner",
    }:
        from memu.workflow import runner as _runner

        return getattr(_runner, name)

    raise AttributeError(f"module 'memu.app' has no attribute {name!r}")
