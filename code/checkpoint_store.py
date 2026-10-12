"""Persistence contracts and SQLite checkpoints for corpus scoring."""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class CheckpointStore(ABC):
    """Persistence contract for resumable classifier scoring."""

    @abstractmethod
    def completed_score_ids(self) -> set[str]:
        """Return sentence identifiers with persisted classifier scores."""

    @abstractmethod
    def load_scores(self) -> dict[str, float]:
        """Return persisted scores indexed by sentence identifier."""

    @abstractmethod
    def save_scores(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Persist classifier scores atomically."""

    @abstractmethod
    def close(self) -> None:
        """Release resources owned by the store."""

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class SQLiteCheckpointStore(CheckpointStore):
    """SQLite checkpoint bound to one corpus and scoring configuration."""

    _SENSITIVE_KEY_FRAGMENTS = ("api_key", "secret", "password", "credential")

    def __init__(
        self,
        path: Path,
        corpus_hash: str,
        configuration: Mapping[str, str],
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._validate_configuration(configuration)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        self._validate_or_initialize_metadata(corpus_hash, configuration)

    @classmethod
    def _validate_configuration(cls, configuration: Mapping[str, str]) -> None:
        sensitive = [
            str(key)
            for key in configuration
            if any(
                fragment in str(key).lower()
                for fragment in cls._SENSITIVE_KEY_FRAGMENTS
            )
        ]
        if sensitive:
            raise ValueError(
                "Checkpoint configuration must not contain secrets: "
                + ", ".join(sorted(sensitive))
            )

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scores (
                sentence_id TEXT PRIMARY KEY,
                hate_score REAL NOT NULL CHECK (hate_score >= 0 AND hate_score <= 1)
            );
            """
        )

    def _validate_or_initialize_metadata(
        self,
        corpus_hash: str,
        configuration: Mapping[str, str],
    ) -> None:
        expected_configuration = dict(configuration)
        expected = {
            "corpus_hash": str(corpus_hash),
            "configuration": json.dumps(
                expected_configuration,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        existing = dict(self._connection.execute("SELECT key, value FROM metadata"))
        if not existing:
            with self._connection:
                self._connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items()
                )
            return
        if existing.get("corpus_hash") != expected["corpus_hash"]:
            raise ValueError("Checkpoint corpus fingerprint does not match input.")

        try:
            existing_configuration = json.loads(existing["configuration"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Checkpoint configuration is invalid.") from error
        matching_configuration = all(
            existing_configuration.get(key) == value
            for key, value in expected_configuration.items()
        )
        if not matching_configuration:
            raise ValueError("Checkpoint configuration does not match this run.")
        if existing["configuration"] != expected["configuration"]:
            with self._connection:
                self._connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'configuration'",
                    (expected["configuration"],),
                )

    def completed_score_ids(self) -> set[str]:
        with self._lock:
            return {
                str(row[0])
                for row in self._connection.execute("SELECT sentence_id FROM scores")
            }

    def load_scores(self) -> dict[str, float]:
        with self._lock:
            return {
                str(sentence_id): float(score)
                for sentence_id, score in self._connection.execute(
                    "SELECT sentence_id, hate_score FROM scores"
                )
            }

    def save_scores(self, rows: Iterable[Mapping[str, Any]]) -> None:
        values = [(str(row["sentence_id"]), float(row["hate_score"])) for row in rows]
        if not values:
            return
        with self._lock, self._connection:
            self._connection.executemany(
                """
                INSERT INTO scores(sentence_id, hate_score) VALUES (?, ?)
                ON CONFLICT(sentence_id) DO UPDATE SET hate_score = excluded.hate_score
                """,
                values,
            )

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            with self._lock:
                connection.close()
                self._connection = None
