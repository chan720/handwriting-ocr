"""Configuration loading: YAML file + command-line overrides -> nested dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BackendConfig:
    type: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: str = "local"
    model: str = "qwen3-vl:8b"
    timeout: float = 900
    max_tokens: int = 4096
    temperature: float = 0.0
    repetition_penalty: float = 1.05
    num_ctx: int = 16384
    num_thread: int = 0
    num_gpu: int = -1
    disable_thinking: bool = True
    extra_body: dict = field(default_factory=dict)


@dataclass
class TransformersConfig:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device_map: str = "auto"
    dtype: str = "auto"
    load_in_4bit: bool = False
    attn_implementation: str | None = None


@dataclass
class IngestConfig:
    pdf_dpi: int = 200


@dataclass
class PreprocessConfig:
    max_side: int = 2048
    min_side: int = 1200
    deskew: bool = True
    enhance_contrast: bool = False
    denoise: bool = False
    rotate: int = 0


@dataclass
class TilingConfig:
    mode: str = "fallback"
    strips: int = 2
    overlap: float = 0.08


@dataclass
class RefineConfig:
    enabled: bool = False


@dataclass
class PromptConfig:
    language: str = ""
    right_to_left: bool = False
    context: str = ""


@dataclass
class OutputConfig:
    formats: list = field(default_factory=lambda: ["txt", "md", "json"])
    page_separator: str = "\n\n"
    save_debug_images: bool = False


@dataclass
class RuntimeConfig:
    workers: int = 1


@dataclass
class Config:
    backend: BackendConfig = field(default_factory=BackendConfig)
    transformers: TransformersConfig = field(default_factory=TransformersConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _fill(dc, data: dict[str, Any], path: str = ""):
    known = {f.name: f for f in fields(dc)}
    for key, value in (data or {}).items():
        if key not in known:
            raise ValueError(f"Unknown config key: {path}{key}")
        current = getattr(dc, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config section {path}{key} must be a mapping")
            _fill(current, value, f"{path}{key}.")
        else:
            setattr(dc, key, value)
    return dc


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load config from YAML (optional) and apply dotted overrides like {"backend.model": "x"}."""
    cfg = Config()
    if path:
        with open(path, encoding="utf-8") as fh:
            _fill(cfg, yaml.safe_load(fh) or {})
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        *parents, leaf = dotted.split(".")
        node = cfg
        for p in parents:
            node = getattr(node, p)
        if not hasattr(node, leaf):
            raise ValueError(f"Unknown config key: {dotted}")
        setattr(node, leaf, value)
    return cfg
