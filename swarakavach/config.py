"""Central configuration: audio params, fusion weights, risk policy thresholds."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"

# ---- Audio ----
SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)   # 16000
CHUNK_OVERLAP_SAMPLES = SAMPLE_RATE // 2           # 0.5s

# ---- Detector layer weights (late fusion; renormalized over available layers) ----
# Bootstrap-calibrated on the demo corpus; recalibrate on ASVspoof5 dev
# (see docs/architecture.md#calibration).
LAYER_WEIGHTS = {
    "artifact": 0.50,   # Detector A: LFCC artifact CNN   — is this synthetic?
    "ssl": 0.28,        # Detector B: wav2vec2 classifier — is this synthetic?
    "speaker": 0.05,    # Detector C: voiceprint match    — is this the claimed person?
    "context": 0.09,    # Detector D: call metadata       — is this call suspicious?
    "intent": 0.08,     # Detector E: transcript rules    — is this request dangerous?
}

# ---- Detector B (SSL) ----
# Validated empirically (see scripts/verify_ssl_mapping.py): human → p(AIVoice)≈0.03,
# TTS → p(AIVoice)≈0.98. Labels: 0=AIVoice (spoof), 1=HumanVoice (bonafide).
SSL_MODEL_ID = os.environ.get("SWK_SSL_MODEL", "Hemgg/Deepfake-audio-detection")
SSL_ENABLED = os.environ.get("SWK_SSL_ENABLED", "1") == "1"
SSL_SPOOF_IDX = int(os.environ.get("SWK_SSL_SPOOF_IDX", "-1"))  # -1 = auto from id2label

# ---- Detector A (artifact) ----
ARTIFACT_CKPT = MODELS_DIR / "artifact_bootstrap.pt"
LFCC_COEFFS = 20

# ---- Risk engine ----
TEMPERATURE = 1.3      # calibration: >1 softens fused probability toward 0.5
EMA_ALPHA = 0.35       # per-chunk exponential smoothing of the fused risk
RISK_LEVELS = [        # (min_risk, level) — first match wins, checked top-down
    (0.65, "CRITICAL"),
    (0.45, "HIGH"),
    (0.25, "MID"),
    (0.00, "LOW"),
]
ACTIONS = {
    "LOW": [],
    "MID": ["CAUTION_PROMPT", "STEP_UP_MFA"],
    "HIGH": ["HOLD_TRANSACTION", "SUPERVISOR_ALERT", "LIVENESS_CHALLENGE"],
    "CRITICAL": ["HOLD_TRANSACTION", "SUPERVISOR_ALERT", "LIVENESS_CHALLENGE", "INCIDENT_WEBHOOK"],
}

# ---- Speaker consistency calibration ----
# ponytail: bootstrap-calibrated constants; recalibrate on real enrollment data
SPEAKER_COS_REF = 0.92   # cosine above this = consistent with enrollment
SPEAKER_COS_SPAN = 0.15  # cosine this far below ref = fully inconsistent

# ---- Alerts & audit ----
WEBHOOK_URL = os.environ.get("SWK_WEBHOOK_URL", "")
AUDIT_LEDGER = os.environ.get("SWK_AUDIT_LEDGER", str(DATA_DIR / "audit" / "ledger.jsonl"))
