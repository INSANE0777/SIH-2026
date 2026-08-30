"""Layer E — conversation-intent risk: rule-based social-engineering signals.

Answers "is what they are asking for dangerous?" — independent of whether the
voice is synthetic. Input is a call transcript (from ASR in production; passed
explicitly by the API/demo caller here). Bilingual keyword rules (English +
Hindi, romanized and Devanagari). Deliberately simple, auditable and
explainable: every score comes with the list of matched rules.
"""
import re

RULES = {
    "fund_transfer": {
        "weight": 0.30,
        "keywords": ("transfer", "send money", "rupees", "lakh", "crore", "imps",
                     "neft", "upi", "payment", "approve the", "paise",
                     "रुपये", "लाख", "ट्रांसफर", "स्थानांतरण", "पैसे"),
    },
    "credentials": {
        "weight": 0.35,
        "keywords": ("otp", "password", "pin", "cvv", "aadhaar", "verification code",
                     "पासवर्ड", "ओटीपी"),
    },
    "urgency": {
        "weight": 0.20,
        "keywords": ("immediately", "urgent", "right now", "quickly", "don't waste",
                     "turant", "jaldi", "तुरंत", "जल्दी"),
    },
    "secrecy": {
        "weight": 0.25,
        "keywords": ("don't tell", "do not tell", "don't discuss", "confidential",
                     "keep this between", "kisi se mat", "kisi ko mat",
                     "किसी से न", "किसी को न", "गोपनीय"),
    },
    "authority": {
        "weight": 0.20,
        "keywords": ("regional manager", "manager", "ceo", "director", "police",
                     "customs", "income tax", "cbi", "official", "bank officer",
                     "प्रबंधक", "पुलिस", "अधिकारी"),
    },
}


def matched_rules(transcript: str) -> list[str]:
    """Which social-engineering rule categories the transcript triggers."""
    if not transcript:
        return []
    t = transcript.lower()
    return [name for name, r in RULES.items()
            if any(kw in t for kw in r["keywords"])]


def score(transcript: str | None) -> float | None:
    """Intent risk in [0, 1] from the transcript; None when no transcript.

    ponytail: keyword rules, no NLP model — auditable and demo-honest; swap in
    a fine-tuned IndicBERT classifier if keyword precision becomes the ceiling.
    """
    if not transcript:
        return None
    total = sum(RULES[name]["weight"] for name in matched_rules(transcript))
    return min(total, 1.0)


def explain(transcript: str | None) -> dict:
    """Explainable payload: score + matched categories + matched keywords."""
    if not transcript:
        return {"score": None, "matched": [], "keywords": []}
    t = transcript.lower()
    kws = sorted({kw for r in RULES.values() for kw in r["keywords"] if kw in t})
    return {"score": score(transcript), "matched": matched_rules(transcript),
            "keywords": kws}
