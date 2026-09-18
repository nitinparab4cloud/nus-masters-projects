# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/05-governed-rag-agent).
# See this project's LICENSE file for reuse terms.

"""
audit_log.py
------------
An append-only, hash-chained audit log. Every entry's hash covers its own
content plus the previous entry's hash, so altering or deleting a past
entry breaks every hash after it -- the "immutable-style audit trail" the
employer-skills document asks for, done honestly: this is tamper-evident
within a single process's memory, not a blockchain or a real WORM store,
and the README says so plainly. What it demonstrates is the *pattern*,
which is the same pattern a real evidence pipeline would build on top of
an actual append-only store.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field

GENESIS_HASH = "0" * 64


@dataclass
class AuditEntry:
    seq: int
    timestamp: float
    event_type: str
    payload: dict
    prev_hash: str
    entry_hash: str


def _hash_entry(seq: int, timestamp: float, event_type: str, payload: dict, prev_hash: str) -> str:
    body = json.dumps(
        {"seq": seq, "timestamp": timestamp, "event_type": event_type, "payload": payload, "prev_hash": prev_hash},
        sort_keys=True,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass
class AuditLog:
    entries: list = field(default_factory=list)
    _clock: callable = field(default=time.time, repr=False)

    def append(self, event_type: str, payload: dict) -> AuditEntry:
        seq = len(self.entries)
        prev_hash = self.entries[-1].entry_hash if self.entries else GENESIS_HASH
        timestamp = self._clock()
        entry_hash = _hash_entry(seq, timestamp, event_type, payload, prev_hash)
        entry = AuditEntry(
            seq=seq, timestamp=timestamp, event_type=event_type,
            payload=payload, prev_hash=prev_hash, entry_hash=entry_hash,
        )
        self.entries.append(entry)
        return entry

    def verify_chain(self) -> "ChainVerification":
        expected_prev = GENESIS_HASH
        for entry in self.entries:
            if entry.prev_hash != expected_prev:
                return ChainVerification(valid=False, broken_at_seq=entry.seq, reason="prev_hash mismatch")
            recomputed = _hash_entry(entry.seq, entry.timestamp, entry.event_type, entry.payload, entry.prev_hash)
            if recomputed != entry.entry_hash:
                return ChainVerification(valid=False, broken_at_seq=entry.seq, reason="entry_hash mismatch")
            expected_prev = entry.entry_hash
        return ChainVerification(valid=True, broken_at_seq=None, reason=None)

    def events_of_type(self, event_type: str) -> list:
        return [e for e in self.entries if e.event_type == event_type]


@dataclass
class ChainVerification:
    valid: bool
    broken_at_seq: int | None
    reason: str | None
