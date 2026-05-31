import sqlite3
import threading
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_RETRY_SLEEP = 0.05
_MAX_RETRIES = 5


class DatabaseManager:
    """Shared SQLite connection manager.

    Replaces 7 independent SQLite databases with a single connection.
    All tables coexist in one database file with WAL journal mode and
    foreign-key enforcement enabled.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._init_connection()

    def _init_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=3000")
        conn.row_factory = sqlite3.Row
        self._local.conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._init_connection()
        return self._local.conn

    def execute(self, sql: str, params=()):
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self.conn.execute(sql, params)
            except sqlite3.OperationalError as e:
                if "locked" in str(e) or "busy" in str(e):
                    last_error = e
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(_RETRY_SLEEP * (2 ** attempt))
                        continue
                raise
        raise last_error

    def commit(self):
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self.conn.commit()
            except sqlite3.OperationalError as e:
                if "locked" in str(e) or "busy" in str(e):
                    last_error = e
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(_RETRY_SLEEP * (2 ** attempt))
                        continue
                raise
        raise last_error

    def close(self):
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


_db_instance: Optional[DatabaseManager] = None
_db_instances: Dict[str, DatabaseManager] = {}
_lock = threading.Lock()


def escape_like(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped


def get_database(db_path: Optional[str] = None) -> DatabaseManager:
    global _db_instance

    if db_path is None:
        if _db_instance is not None:
            return _db_instance
        with _lock:
            if _db_instance is not None:
                return _db_instance
            from config import config
            db_path = getattr(config, "DATABASE_PATH", "qasystem.db")
            _db_instance = DatabaseManager(db_path)
            return _db_instance

    # Explicit path requested (e.g. by tests) — return a dedicated instance
    if db_path in _db_instances:
        return _db_instances[db_path]
    with _lock:
        if db_path in _db_instances:
            return _db_instances[db_path]
        instance = DatabaseManager(db_path)
        _db_instances[db_path] = instance
        return instance
