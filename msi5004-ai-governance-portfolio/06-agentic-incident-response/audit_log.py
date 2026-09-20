# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/06-agentic-incident-response).
# See this project's LICENSE file for reuse terms.

"""
audit_log.py

A SHA-256 hash-chained, append-only audit trail. Same pattern as Case 05
(05-governed-rag-agent/audit_log.py): each entry's hash covers its own
content plus the previous entry's hash, so altering or deleting a past
entry breaks every hash after it.

Honest about what this is: tamper-evident within one process's memory,
not a real WORM store, not a blockchain, and not resistant to someone with
write access to the process's own memory. What it demonstrates is the
*pattern* -- every governed action leaves a trail that can be independently
verified -- which is the control that matters for "detected but not
escalated" (row 4 of the incident's control mapping): you cannot escalate
what you never durably recorded in the first place.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


GENESIS_HASH = "0" * 64


def _hash_entry(seq: int, event: str, payload: dict, prev_hash: str, ts: float) -> str:
    canonical = json.dumps(
        {"seq": seq, "event": event, "payload": payload, "prev_hash": prev_hash, "ts": ts},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class AuditEntry:
    seq: int
    event: str
    payload: dict
    prev_hash: str
    ts: float
    entry_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.entry_hash = _hash_entry(self.seq, self.event, self.payload, self.prev_hash, self.ts)

    def as_dict(self) -> dict:
        return {
            "seq": self.seq,
            "event": self.event,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "ts": self.ts,
            "entry_hash": self.entry_hash,
        }


class AuditLog:
    """Append-only, hash-chained log. `append` is the only write path."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def append(self, event: str, payload: dict | None = None, ts: float | None = None) -> AuditEntry:
        payload = payload or {}
        prev_hash = self._entries[-1].entry_hash if self._entries else GENESIS_HASH
        entry = AuditEntry(
            seq=len(self._entries),
            event=event,
            payload=payload,
            prev_hash=prev_hash,
            ts=ts if ts is not None else time.time(),
        )
        self._entries.append(entry)
        return entry

    def entries(self) -> list[dict]:
        return [e.as_dict() for e in self._entries]

    def verify_chain(self) -> tuple[bool, str]:
        """Recompute every hash from scratch; the first mismatch names the break."""
        prev_hash = GENESIS_HASH
        for e in self._entries:
            expected = _hash_entry(e.seq, e.event, e.payload, prev_hash, e.ts)
            if expected != e.entry_hash:
                return False, f"chain broken at seq {e.seq}: stored hash does not match recomputed hash"
            if e.prev_hash != prev_hash:
                return False, f"chain broken at seq {e.seq}: prev_hash does not match the prior entry"
            prev_hash = e.entry_hash
        return True, f"chain intact: {len(self._entries)} entries verified"

    def __len__(self) -> int:
        return len(self._entries)
