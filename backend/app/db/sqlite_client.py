"""SQLAlchemy 2.0 SQLite transactional state database (WAL mode)."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


class SQLiteDatabase:
    def __init__(self, db_path: Path | str | None = None, initial_cash: float | None = None) -> None:
        settings = get_settings()
        self.db_path = Path(db_path) if db_path is not None else Path(settings.sqlite_path)
        self._default_initial_cash = float(initial_cash) if initial_cash is not None else float(settings.paper_initial_cash)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(
            f"sqlite:///{self.db_path.as_posix()}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        event.listen(self.engine, "connect", _set_sqlite_pragmas)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
            class_=Session,
        )
        self.init_schema()

    def init_schema(self) -> None:
        from app.db import models  # noqa: F401

        Base.metadata.create_all(bind=self.engine)
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        from app.db.models import BrokerConfig, PaperAccount, RiskConfig

        settings = get_settings()
        init_cash = self._default_initial_cash
        with self.session_scope() as session:
            if session.get(PaperAccount, "default_paper") is None:
                session.add(
                    PaperAccount(
                        account_id="default_paper",
                        name="A-Share Alpha 默认模拟账户",
                        broker_mode="PAPER",
                        initial_cash=init_cash,
                        cash=init_cash,
                        total_equity=init_cash,
                        peak_equity=init_cash,
                        current_drawdown=0.0,
                        fuse_status="NORMAL",
                    )
                )
            if session.get(RiskConfig, 1) is None:
                session.add(RiskConfig(id=1))
            if session.get(BrokerConfig, 1) is None:
                session.add(
                    BrokerConfig(
                        id=1,
                        active_mode="PAPER",
                        wecom_webhook_url=settings.wecom_webhook_url,
                        dingtalk_webhook_url=settings.dingtalk_webhook_url,
                        dingtalk_secret=settings.dingtalk_secret,
                        qmt_userdata_path=settings.qmt_userdata_path,
                        qmt_account_id=settings.qmt_account_id,
                        qmt_dry_run=settings.qmt_dry_run,
                        tushare_token_configured=bool(settings.tushare_token),
                    )
                )

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        self.engine.dispose()


_sqlite_singleton: SQLiteDatabase | None = None
_sqlite_lock = threading.Lock()


def get_sqlite_db() -> SQLiteDatabase:
    global _sqlite_singleton
    with _sqlite_lock:
        if _sqlite_singleton is None:
            _sqlite_singleton = SQLiteDatabase()
        return _sqlite_singleton


def reset_sqlite_singleton() -> None:
    global _sqlite_singleton
    with _sqlite_lock:
        if _sqlite_singleton is not None:
            _sqlite_singleton.close()
            _sqlite_singleton = None
