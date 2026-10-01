"""Loads versioned prompts from prompts/registry.json + prompts/*.md.

See DESIGN.md section 5. Prompts are kept as first-class, versioned files
rather than hardcoded strings so every eval run can stamp exactly which
prompt version it used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class PromptVersion:
    id: str
    version: str
    text: str
    compatible_models: list[str]
    notes: str


def _load_registry(prompts_dir: Path) -> dict:
    registry_path = prompts_dir / "registry.json"
    return json.loads(registry_path.read_text(encoding="utf-8"))


def load_prompt(prompt_id: str, prompts_dir: Path = PROMPTS_DIR) -> PromptVersion:
    registry = _load_registry(prompts_dir)
    if prompt_id not in registry:
        raise KeyError(f"Unknown prompt_id {prompt_id!r}. Known: {sorted(registry)}")
    entry = registry[prompt_id]
    text = (prompts_dir / entry["file"]).read_text(encoding="utf-8")
    return PromptVersion(
        id=prompt_id,
        version=entry["version"],
        text=text,
        compatible_models=entry.get("compatible_models", []),
        notes=entry.get("notes", ""),
    )


def list_prompts(prompts_dir: Path = PROMPTS_DIR) -> list[PromptVersion]:
    registry = _load_registry(prompts_dir)
    return [load_prompt(pid, prompts_dir) for pid in registry]
