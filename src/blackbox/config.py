"""Configuration: `blackbox.toml`, then `BLACKBOX__*` environment variables, then `.env`.

Environment variables use `__` between levels, e.g. `BLACKBOX__SERVER__PORT=8300`. Validation fails at startup on
bad values.
"""

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

_config_path: ContextVar[Path | None] = ContextVar("blackbox_config_path", default=None)


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8200, ge=0, le=65535)
    public_url: str | None = None  # how links are printed; defaults to http://host:port

    @property
    def base_url(self) -> str:
        return self.public_url or f"http://{self.host}:{self.port}"


class StoreConfig(BaseModel):
    path: Path = Path("data/blackbox.db")


class OllamaConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:4b"
    timeout_seconds: float = Field(default=300, gt=0)


class RunsConfig(BaseModel):
    quiet_seconds: float = Field(default=5, ge=0)
    orphan_seconds: float = Field(default=60, ge=0)  # complete a run with no identifiable root after this long
    keep_unmatched: bool = False
    tick_seconds: float = Field(default=1.0, gt=0)


class JudgesConfig(BaseModel):
    judge_replays: bool = True  # fidelity reports judge the replay too (exact replays hit the cache: no model call)


class ClustersConfig(BaseModel):
    embedder: Literal["fastembed", "hashing"] = "fastembed"  # hashing: no model download (tests, offline machines)
    model: str = "BAAI/bge-small-en-v1.5"
    cache_dir: Path = Path("data/models")
    min_cluster_size: int = Field(default=3, ge=2)
    onehot_weight: float = Field(default=0.5, ge=0)  # weight of ending, category, flags and node next to the text
    recluster_after: int = Field(default=10, ge=1)  # unclustered failures waiting before a re-clustering
    rename_change: float = Field(default=0.3, ge=0, le=1)  # regenerate a name when members change by more


class SamplingConfig(BaseModel):
    judge_rate: float = Field(default=0.3, ge=0, le=1)  # share of ordinary runs that trusted judges score
    always_judge_flagged: bool = True  # runs with a metric flag or a failure ending are always judged
    max_judge_calls_per_hour: int = Field(default=60, ge=0)


class RateDropConfig(BaseModel):
    """Rate drops: a Bernoulli CUSUM over the current window against a lagged baseline (see `live.detectors`)."""

    baseline_runs: int = Field(default=100, ge=1)
    min_baseline: int = Field(default=30, ge=1)
    window: int = Field(default=20, ge=1)
    min_current: int = Field(default=8, ge=1)
    design_drop: float = Field(default=0.3, gt=0, lt=1)  # the drop the test is tuned to catch quickly
    threshold: float = Field(default=6.0, gt=0)  # h, in nats of evidence
    delta: float = Field(default=0.15, ge=0, lt=1)  # reported: P(current rate < baseline - delta)


class ShiftConfig(BaseModel):
    k: float = Field(default=0.5, ge=0)
    h: float = Field(default=5.0, gt=0)
    z_cap: float = Field(default=4.0, gt=0)  # one outlier alone can't cross h


class NewModeConfig(BaseModel):
    minutes: float = Field(default=30, gt=0)
    growth: int = Field(default=3, ge=1)


class LiveConfig(BaseModel):
    sampling: dict[str, SamplingConfig] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=lambda: ["live", "traffic"])  # runs the detectors watch
    model_lane_max_wait_seconds: float = Field(default=600, ge=0)
    rate_drop: RateDropConfig = Field(default_factory=RateDropConfig)
    shift: ShiftConfig = Field(default_factory=ShiftConfig)
    new_mode: NewModeConfig = Field(default_factory=NewModeConfig)
    clear_evaluations: int = Field(default=2, ge=1)
    cooldown_minutes: float = Field(default=30, ge=0)
    patch_minutes: float = Field(default=30, gt=0)

    def sampling_for(self, profile: str) -> SamplingConfig:
        return self.sampling.get(profile) or SamplingConfig()


class AlertsConfig(BaseModel):
    """Telegram: a separate bot from PaperPilot's. The token is a secret: never logged, never stored."""

    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None
    telegram_api: str = "https://api.telegram.org"


class UpstreamConfig(BaseModel):
    name: str
    listen_port: int = Field(ge=0, le=65535)
    target: str
    record_paths: list[str] = Field(default_factory=lambda: ["/*"])
    timeout_seconds: float = Field(default=600, gt=0)
    stateful: bool = False

    @field_validator("target")
    @classmethod
    def _no_localhost(cls, value: str) -> str:
        if "//localhost" in value:
            raise ValueError("use 127.0.0.1, not localhost (Windows tries ::1 first and loses ~2 s per call)")
        return value.rstrip("/")


class ProxyConfig(BaseModel):
    enabled: bool = True
    upstreams: list[UpstreamConfig] = Field(default_factory=list)
    redact_headers: list[str] = Field(
        default_factory=lambda: ["authorization", "x-api-key", "api-key", "cookie", "set-cookie", "proxy-authorization"]
    )
    session_ttl_seconds: float = Field(default=1800, gt=0)

    def upstream(self, name: str) -> UpstreamConfig:
        for upstream in self.upstreams:
            if upstream.name == name:
                return upstream
        raise KeyError(f"no proxy upstream named {name!r}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BLACKBOX__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    runs: RunsConfig = Field(default_factory=RunsConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    judges: JudgesConfig = Field(default_factory=JudgesConfig)
    clusters: ClustersConfig = Field(default_factory=ClustersConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)  # per-profile options, e.g. base_url
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        path = _config_path.get()
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        if path is not None and path.exists():
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=path))
        return tuple(sources)


def default_config_path() -> Path:
    return Path(os.environ.get("BLACKBOX_CONFIG", "blackbox.toml"))


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from `path` (default: `$BLACKBOX_CONFIG` or `./blackbox.toml`), the environment and `.env`."""
    token = _config_path.set(path if path is not None else default_config_path())
    try:
        return Settings()
    finally:
        _config_path.reset(token)
