"""Global application configuration using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    duckdb_path: Path | None = None
    sqlite_path: Path | None = None
    models_dir: Path | None = None

    data_primary_provider: str = "akshare"
    data_backup_provider: str = "baostock"
    tushare_token: str = ""

    scheduler_enabled: bool = True
    paper_initial_cash: float = 1_000_000.0
    intraday_refresh_seconds: int = 120

    wecom_webhook_url: str = ""
    dingtalk_webhook_url: str = ""
    dingtalk_secret: str = ""

    qmt_userdata_path: str = ""
    qmt_account_id: str = ""
    qmt_dry_run: bool = True

    def resolve_paths(self) -> "Settings":
        data_dir = self.data_dir
        if not data_dir.is_absolute():
            data_dir = (BACKEND_ROOT / data_dir).resolve()
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if self.duckdb_path is None:
            self.duckdb_path = self.data_dir / "market_analytics.duckdb"
        elif not self.duckdb_path.is_absolute():
            self.duckdb_path = (BACKEND_ROOT / self.duckdb_path).resolve()

        if self.sqlite_path is None:
            self.sqlite_path = self.data_dir / "studio_state.db"
        elif not self.sqlite_path.is_absolute():
            self.sqlite_path = (BACKEND_ROOT / self.sqlite_path).resolve()

        if self.models_dir is None:
            self.models_dir = self.data_dir / "models"
        elif not self.models_dir.is_absolute():
            self.models_dir = (BACKEND_ROOT / self.models_dir).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings().resolve_paths()
