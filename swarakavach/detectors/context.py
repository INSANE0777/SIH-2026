"""Layer D — call-context risk engine.

Non-audio signals: caller-ID binding, known-contact status, transaction size,
origin, velocity, prior fraud flags. Pure deterministic rules — auditable by
design, which bank fraud teams require.
"""
from .. import config


def score(context: dict | None):
    """Context risk in [0,1] from call metadata; None when no context is provided."""
    if not context:
        return None
    c = {str(k).lower(): v for k, v in context.items()}
    r = 0.0

    def flag(name):  # treat "true"/True/1 as set
        return str(c.get(name, "")).lower() in ("true", "1", "yes")

    if "caller_id_match" in c and not flag("caller_id_match"):
        r += 0.30
    if "enrolled_contact" in c and not flag("enrolled_contact"):
        r += 0.15
    if flag("known_fraud_flag"):
        r += 0.40
    if flag("intl_origin"):
        r += 0.15
    try:
        tx = float(c.get("tx_amount", 0) or 0)
        if tx > 0:
            r += min(tx / 5_000_000.0, 0.25)  # scales up to ₹50L
    except (TypeError, ValueError):
        pass
    try:
        if int(c.get("calls_today", 0) or 0) > 3:
            r += 0.15
    except (TypeError, ValueError):
        pass
    return float(min(r, 1.0))
