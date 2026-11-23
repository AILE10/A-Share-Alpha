"""Shared pytest fixtures. Integration tests read the REAL DuckDB store populated by
scripts/m1_verify_real_data.py and are skipped (never faked) when it is empty."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import get_settings  # noqa: E402
from app.db.duckdb_client import get_duckdb  # noqa: E402


@pytest.fixture(scope="session")
def real_db():
    db = get_duckdb()
    n = db.query_scalar("SELECT COUNT(DISTINCT ts_code) FROM daily_bars")
    if not n or n < 20:
        pytest.skip("Real DuckDB store empty - run scripts/m1_verify_real_data.py first")
    return db


@pytest.fixture(scope="session")
def sample_codes(real_db):
    f = get_settings().data_dir / "sample_universe.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return real_db.query_df("SELECT DISTINCT ts_code FROM daily_bars ORDER BY 1")["ts_code"].tolist()


@pytest.fixture(scope="session")
def real_panel(real_db, sample_codes):
    from app.factors.panel import load_market_panel

    return load_market_panel(sample_codes, db=real_db)


@pytest.fixture(scope="session")
def factor_run(real_db, real_panel):
    from app.factors.service import FactorEngine

    return FactorEngine(real_db).build(panel=real_panel)
