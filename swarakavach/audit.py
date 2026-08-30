"""Tamper-evident incident ledger: hash-chained audit records (blockchain-lite).

Every HIGH/CRITICAL verdict appends an incident record to an append-only JSONL
ledger. Each record embeds the previous record's SHA-256 hash, so any after-
the-fact edit breaks the chain and verify_chain() detects it.

Privacy: records store metadata ONLY — risk score, policy triggered, action,
model version, supervisor decision, timestamp. Never audio, never transcripts.
"""
import hashlib
import json
import time
from pathlib import Path

from . import config


def _hash(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def ledger_path(path=None) -> Path:
    return Path(path or config.AUDIT_LEDGER)


def read(path=None) -> list[dict]:
    p = ledger_path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def append(record: dict, path=None) -> dict:
    """Append a record; returns it with index/prev_hash/hash filled in."""
    p = ledger_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    records = read(p)
    rec = {"index": len(records), "ts": round(time.time(), 3), **record}
    rec["prev_hash"] = records[-1]["hash"] if records else "GENESIS"
    rec["hash"] = _hash({k: v for k, v in rec.items() if k != "hash"})
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def verify_chain(path=None) -> tuple[bool, int]:
    """Recompute the chain; returns (valid, first_broken_index)."""
    records = read(path)
    prev = "GENESIS"
    for i, rec in enumerate(records):
        body = {k: v for k, v in rec.items() if k != "hash"}
        if rec.get("index") != i or rec.get("prev_hash") != prev \
                or _hash(body) != rec.get("hash"):
            return False, i
        prev = rec["hash"]
    return True, -1


def incident(session: dict, model_version: str, path=None) -> dict:
    """Build + append the standard incident record for a session verdict."""
    v = session["verdict"]
    return append({
        "type": "incident",
        "incident_id": f"INC-{session['session_id']}",
        "session_id": session["session_id"],
        "level": v["level"],
        "risk_peak": v["risk_peak"],
        "model_version": model_version,
        "policy_triggered": v["actions"],
        "transaction_action": "HELD" if "HOLD_TRANSACTION" in v["actions"] else "PROCEEDED",
        "supervisor_decision": "PENDING",
    }, path)


def decision(incident_id: str, decision: str, supervisor: str, path=None) -> dict:
    """Append a supervisor decision record referencing an incident."""
    return append({
        "type": "decision",
        "incident_id": incident_id,
        "decision": decision,
        "supervisor": supervisor,
    }, path)
