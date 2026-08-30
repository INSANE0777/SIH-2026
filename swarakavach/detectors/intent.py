"""Layer E — conversation-intent risk: rule-based social-engineering signals.

Answers "is what they are asking for dangerous?" - independent of whether the
voice is synthetic. Input is a call transcript - from live ASR (swarakavach/
asr.py) in production, or passed explicitly by API/demo callers. Bilingual
keyword rules (English + Hindi, romanized and Devanagari).

ASR output is noisy (TTS-warped Hindi comes back as a mangled word), so
Devanagari keywords match FUZZILY (edit distance <= 1 per token) - exact
substring matching silently missed the Hindi attacker in testing. Every score
carries the matched rules (auditable by design). Swap in a fine-tuned
IndicBERT classifier if keyword precision becomes the ceiling.
"""
import re


def _devanagari_norm(s: str) -> str:
    """Normalize Devanagari for fuzzy comparison: strip trailing/inherent
    virama marks, unify homophones (anusvara ~ n, sibilants, ba/va) —
    TTS+ASR output like 'प्रभंदक'/'ट्रान्स्फर'/'लाक्' still matches
    'प्रबंधक'/'ट्रांसफर'/'लाख' after this transform."""
    s = (s.replace("्", "")            # virama (halant)
          .replace("ं", "न")      # anusvara -> n
          .replace("ष", "स")      # ssa -> sa
          .replace("श", "स")      # sha -> sa
          .replace("व", "ब"))     # va -> ba (common ASR swap)
    return s


def _fuzzy_word_in(word: str, text: str) -> bool:
    """True if `word` appears in `text` allowing <= max_dist edits per token.
    TTS+ASR mangle Devanagari inflections; exact matching missed real attacks."""
    if word in text:
        return True
    # Devanagari words are long; 2 edits for >=6-char words stays specific
    max_dist = 2 if len(word) >= 6 else 1

    def lev(a: str, b: str) -> int:
        if abs(len(a) - len(b)) > max_dist:
            return max_dist + 1
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    word = _devanagari_norm(word)
    for tok in text.split():
        w = _devanagari_norm(tok.strip("।,!?;:'\""))
        if abs(len(w) - len(word)) <= max_dist and lev(word, w) <= max_dist:
            return True
    return False

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
    """Which rule categories the transcript triggers. Short ASCII keywords
    (e.g. 'otp', 'pin', 'cbi') match whole-word only; Devanagari keywords
    match fuzzily against ASR-noisy text."""
    if not transcript:
        return []
    t = re.sub(r"\s+", " ", transcript.lower()).strip()

    def hit(rule) -> bool:
        for kw in rule["keywords"]:
            if kw.isascii():
                # ASCII: short alpha keywords word-boundary matched; phrases substring
                if kw.isalpha() and len(kw) <= 5:
                    if re.search(rf"\b{re.escape(kw)}\b", t):
                        return True
                elif kw in t:
                    return True
            else:
                # Devanagari: fuzzy word match (ASR mangles TTS Hindi)
                if _fuzzy_word_in(kw, t):
                    return True
        return False

    return [name for name, r in RULES.items() if hit(r)]


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
