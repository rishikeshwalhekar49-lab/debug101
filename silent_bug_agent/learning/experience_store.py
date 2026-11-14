"""Agent 6: experience memory.

SQLite keeps every repair attempt, the bandit's statistics and the lessons used in
prompts. ChromaDB holds embeddings of accepted repairs for semantic recall, using
its built-in `all-MiniLM-L6-v2` model, which runs on the CPU and needs no key. If
ChromaDB or its model is unavailable, recall falls back to word overlap and says so.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import data_home

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
    id TEXT PRIMARY KEY, run_id TEXT, created_at REAL, bug_id TEXT, category TEXT, strategy TEXT, source TEXT,
    accepted INTEGER, reward REAL, iterations INTEGER, summary TEXT, diff TEXT, notes TEXT, p_value REAL,
    effect_size REAL, exemplar_ids TEXT, active INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS bandit (context TEXT, strategy TEXT, count REAL, reward_sum REAL, PRIMARY KEY (context, strategy));
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, created_at REAL, body TEXT, source TEXT, active INTEGER DEFAULT 1,
    uses INTEGER DEFAULT 0, wins INTEGER DEFAULT 0);
"""


def _words(text: str) -> set:
    return set(re.findall(r"[a-z_]{3,}", text.lower()))


class ExperienceStore:
    def __init__(self, home: Optional[Path] = None, *, use_vectors: bool = True) -> None:
        self.home = Path(home) if home else data_home()
        try:
            self.home.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self.path = self.home / "experience.sqlite3"
        self.use_vectors = use_vectors
        self.notes: List[str] = []
        self._collection: Any = None
        self._vector_failed = False
        self._lock = threading.Lock()
        self._fallback_db: Optional[sqlite3.Connection] = None
        with closing(self._connect()) as db, db:
            db.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        if self._fallback_db is not None:
            return self._fallback_db
        try:
            db = sqlite3.connect(self.path, timeout=10)
            db.row_factory = sqlite3.Row
            return db
        except Exception as exc:
            self.notes.append(f"memory: SQLite file failed ({exc}), using in-memory fallback")
            self._fallback_db = sqlite3.connect(":memory:", check_same_thread=False)
            self._fallback_db.row_factory = sqlite3.Row
            self._fallback_db.executescript(_SCHEMA)
            return self._fallback_db

    # ---------------------------------------------------------------- vectors
    def _vectors(self) -> Any:
        if not self.use_vectors or self._vector_failed:
            return None
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings

                client = chromadb.PersistentClient(path=str(self.home / "chroma"),
                                                   settings=Settings(anonymized_telemetry=False))
                self._collection = client.get_or_create_collection("repairs")
            except Exception as exc:  # any install or file problem: memory must keep working
                self._vector_failed = True
                self.notes.append(f"memory: ChromaDB unavailable, using word overlap ({type(exc).__name__})")
                return None
        return self._collection

    # -------------------------------------------------------------- trajectories
    def remember(self, record: Dict[str, Any]) -> str:
        """Store one repair attempt; accepted ones become examples for future repairs."""
        item = {"id": uuid.uuid4().hex, "run_id": "", "bug_id": "", "category": "", "strategy": "", "source": "",
                "accepted": False, "reward": 0.0, "iterations": 0, "summary": "", "diff": "", "notes": "",
                "p_value": None, "effect_size": None, "exemplar_ids": [], **record}
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO trajectories (id, run_id, created_at, bug_id, category, strategy, source, accepted, reward,"
                " iterations, summary, diff, notes, p_value, effect_size, exemplar_ids) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item["id"], item["run_id"], time.time(), item["bug_id"], item["category"], item["strategy"], item["source"],
                 int(bool(item["accepted"])), float(item["reward"]), int(item["iterations"]), item["summary"], item["diff"],
                 item["notes"], item["p_value"], item["effect_size"], json.dumps(item["exemplar_ids"])))
        if item["accepted"] and item["diff"]:
            collection = self._vectors()
            if collection is not None:
                try:
                    collection.add(ids=[item["id"]], documents=[f"{item['bug_id']} {item['summary']}\n{item['diff'][:1500]}"],
                                   metadatas=[{"bug_id": item["bug_id"]}])
                except Exception as exc:
                    self._vector_failed = True
                    self.notes.append(f"memory: embedding failed, using word overlap ({type(exc).__name__})")
        return item["id"]

    def _rows(self, where: str = "1=1", args: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        with closing(self._connect()) as db:
            rows = db.execute(f"SELECT * FROM trajectories WHERE {where} ORDER BY created_at", tuple(args)).fetchall()
        return [dict(r) | {"exemplar_ids": json.loads(r["exemplar_ids"] or "[]")} for r in rows]

    def trajectories(self, *, accepted: Optional[bool] = None) -> List[Dict[str, Any]]:
        return self._rows("accepted=?", [int(accepted)]) if accepted is not None else self._rows()

    def count(self) -> int:
        with closing(self._connect()) as db:
            return int(db.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0])

    def recall(self, query: str, *, k: int = 2) -> List[Dict[str, Any]]:
        """The accepted, active repairs most similar to the query."""
        pool = {r["id"]: r for r in self._rows("accepted=1 AND active=1")}
        if not pool:
            return []
        collection = self._vectors()
        if collection is not None:
            try:
                found = collection.query(query_texts=[query], n_results=min(k * 3, len(pool)))
                ranked = [pool[i] for i in found["ids"][0] if i in pool]
                if ranked:
                    return ranked[:k]
            except Exception as exc:
                self._vector_failed = True
                self.notes.append(f"memory: vector search failed, using word overlap ({type(exc).__name__})")
        wanted = _words(query)
        scored = sorted(pool.values(), key=lambda r: len(wanted & _words(f"{r['bug_id']} {r['summary']}")), reverse=True)
        return [r for r in scored if wanted & _words(f"{r['bug_id']} {r['summary']}")][:k]

    def prune(self, ids: Sequence[str]) -> None:
        """Stop using these repairs as examples. The record stays for audit."""
        if not ids:
            return
        with self._lock, closing(self._connect()) as db, db:
            db.executemany("UPDATE trajectories SET active=0 WHERE id=?", [(i,) for i in ids])
        collection = self._vectors()
        if collection is not None:
            try:
                collection.delete(ids=list(ids))
            except Exception as exc:
                self.notes.append(f"memory: could not delete vectors ({type(exc).__name__})")

    # ------------------------------------------------------------------ bandit
    def load_bandit(self, context: str) -> Dict[str, Dict[str, float]]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT strategy, count, reward_sum FROM bandit WHERE context=?", (context,)).fetchall()
        return {r["strategy"]: {"count": r["count"], "reward_sum": r["reward_sum"]} for r in rows}

    def update_bandit(self, context: str, strategy: str, reward: float, count: float = 1.0) -> None:
        with self._lock, closing(self._connect()) as db, db:
            db.execute("INSERT INTO bandit VALUES (?,?,?,?) ON CONFLICT(context, strategy) DO UPDATE SET "
                       "count=count+excluded.count, reward_sum=reward_sum+excluded.reward_sum",
                       (context, strategy, count, reward))

    def replace_bandit(self, stats: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        with self._lock, closing(self._connect()) as db, db:
            db.execute("DELETE FROM bandit")
            for context, arms in stats.items():
                for strategy, item in arms.items():
                    db.execute("INSERT INTO bandit VALUES (?,?,?,?)", (context, strategy, item["count"], item["reward_sum"]))

    # ----------------------------------------------------------------- lessons
    def active_lessons(self, task: str) -> List[str]:
        with closing(self._connect()) as db:
            row = db.execute("SELECT body FROM lessons WHERE task=? AND active=1 ORDER BY id DESC LIMIT 1", (task,)).fetchone()
        return json.loads(row["body"]) if row else []

    def set_lessons(self, task: str, lessons: Sequence[str], source: str) -> None:
        with self._lock, closing(self._connect()) as db, db:
            db.execute("UPDATE lessons SET active=0 WHERE task=?", (task,))
            db.execute("INSERT INTO lessons (task, created_at, body, source) VALUES (?,?,?,?)",
                       (task, time.time(), json.dumps(list(lessons)), source))

    def score_lessons(self, task: str, accepted: bool) -> None:
        with self._lock, closing(self._connect()) as db, db:
            db.execute("UPDATE lessons SET uses=uses+1, wins=wins+? WHERE task=? AND active=1", (int(accepted), task))

    def lesson_versions(self, task: str) -> List[Dict[str, Any]]:
        with closing(self._connect()) as db:
            return [dict(r) for r in db.execute("SELECT * FROM lessons WHERE task=? ORDER BY id", (task,)).fetchall()]

    def rollback_lessons(self, task: str) -> bool:
        versions = self.lesson_versions(task)
        if len(versions) < 2:
            return False
        with self._lock, closing(self._connect()) as db, db:
            db.execute("UPDATE lessons SET active=0 WHERE task=?", (task,))
            db.execute("UPDATE lessons SET active=1 WHERE id=?", (versions[-2]["id"],))
        return True
