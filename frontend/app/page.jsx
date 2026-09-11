"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const LVL_COLORS = { LOW: "#059669", MID: "#b45309", HIGH: "#c2410c", CRITICAL: "#dc2626" };
const riskColor = (v) => (v < 0.25 ? "#059669" : v < 0.45 ? "#b45309" : v < 0.65 ? "#c2410c" : "#dc2626");
const CTX = encodeURIComponent(JSON.stringify({ caller_id_match: "false", enrolled_contact: "true", tx_amount: 5000000 }));

const LAYERS = [
  ["artifact", "A · Spectral artifacts — synthetic?"],
  ["ssl", "B · SSL wav2vec2 — synthetic?"],
  ["speaker", "C · Voiceprint — this person?"],
  ["context", "D · Call context — suspicious call?"],
  ["intent", "E · Intent — dangerous request?"],
];

export default function Dashboard() {
  const [conn, setConn] = useState(false);
  const [health, setHealth] = useState("connecting…");
  const [risk, setRisk] = useState(0);
  const [level, setLevel] = useState("STANDBY");
  const [actions, setActions] = useState([]);
  const [layers, setLayers] = useState({});
  const [traj, setTraj] = useState([]);
  const [log, setLog] = useState([]);
  const [audit, setAudit] = useState(null);
  const [finalV, setFinalV] = useState(null);
  const [asrText, setAsrText] = useState("");
  const [speakerId, setSpeakerId] = useState("");
  const [manualTr, setManualTr] = useState("");
  const [streaming, setStreaming] = useState(false);

  const wsRef = useRef(null);
  const micRef = useRef(null);
  const procRef = useRef(null);
  const srcRef = useRef(null);
  const streamingRef = useRef(false);
  const trajRef = useRef([]);
  const sparkRef = useRef(null);
  const waveRef = useRef(null);
  const waveBuf = useRef(new Array(74).fill(0));
  const logId = useRef(0);

  const addLog = useCallback((s, warn) => {
    setLog((l) => [{ id: logId.current++, time: new Date().toLocaleTimeString(), s, warn }, ...l].slice(0, 200));
  }, []);

  const fetchAudit = useCallback(() => {
    fetch("/v1/audit")
      .then((r) => r.json())
      .then((a) => setAudit(a))
      .catch(() => {});
  }, []);

  const drawSpark = useCallback((traj) => {
    const c = sparkRef.current;
    if (!c) return;
    const g = c.getContext("2d");
    g.clearRect(0, 0, c.width, c.height);
    g.font = "10px JetBrains Mono, monospace";
    [[0.25, "MID"], [0.45, "HIGH"], [0.65, "CRIT"]].forEach(([lv, lb]) => {
      const y = (1 - lv) * c.height;
      g.strokeStyle = "#e2e8f0";
      g.beginPath();
      g.moveTo(0, y);
      g.lineTo(c.width, y);
      g.stroke();
      g.fillStyle = "#94a3b8";
      g.fillText(lb, 4, y - 3);
    });
    if (traj.length < 2) return;
    const xs = (i) => (i / (traj.length - 1)) * c.width;
    const ys = (v) => (1 - v) * c.height;
    const grad = g.createLinearGradient(0, 0, 0, c.height);
    grad.addColorStop(0, "rgba(37,99,235,.22)");
    grad.addColorStop(1, "rgba(37,99,235,0)");
    g.fillStyle = grad;
    g.beginPath();
    g.moveTo(0, c.height);
    traj.forEach((v, i) => g.lineTo(xs(i), ys(v)));
    g.lineTo(c.width, c.height);
    g.closePath();
    g.fill();
    g.strokeStyle = "#2563eb";
    g.lineWidth = 2;
    g.beginPath();
    traj.forEach((v, i) => (i ? g.lineTo(xs(i), ys(v)) : g.moveTo(xs(i), ys(v))));
    g.stroke();
    const last = traj[traj.length - 1];
    g.fillStyle = riskColor(last);
    g.beginPath();
    g.arc(xs(traj.length - 1), ys(last), 3.5, 0, 7);
    g.fill();
  }, []);

  const onEvent = useCallback(
    (m) => {
      setRisk(m.risk);
      setLevel(m.level);
      setActions(m.actions || []);
      setLayers(m.layers || {});
      const nt = [...trajRef.current, m.risk];
      trajRef.current = nt;
      setTraj(nt);
      drawSpark(nt);
      if (m.transcript_delta) setAsrText(`"${m.transcript_delta}"`);
      addLog(
        `chunk ${String(m.chunk).padStart(2, "0")}  risk=${m.risk.toFixed(3)}  ${m.level}${(m.actions || []).length ? "  ! " + m.actions.join(", ") : ""}`,
        (m.actions || []).length > 0
      );
    },
    [addLog, drawSpark]
  );

  const showFinal = useCallback(
    (m) => {
      setFinalV(m);
      setLevel(m.level);
      setActions(m.actions || []);
      addLog(`■ CALL COMPLETE — ${m.level} (${m.risk_final})`, m.level !== "LOW");
      fetchAudit();
    },
    [addLog, fetchAudit]
  );

  const connect = useCallback(() => {
    const tr = manualTr.trim();
    const ws = new WebSocket(
      `ws://127.0.0.1:8000/v1/stream?speaker_id=${encodeURIComponent(speakerId || "")}&context=${CTX}${tr ? "&transcript=" + encodeURIComponent(tr) : ""}`
    );
    wsRef.current = ws;
    ws.onopen = () => {
      setConn(true);
      addLog("connected — stream ready");
    };
    ws.onclose = () => {
      setConn(false);
      if (streamingRef.current) stopMic();
    };
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      m.final ? showFinal(m) : onEvent(m);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [speakerId, manualTr, addLog, onEvent, showFinal]);

  const startMic = useCallback(async () => {
    try {
      micRef.current = micRef.current || new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      srcRef.current = micRef.current.createMediaStreamSource(stream);
      procRef.current = micRef.current.createScriptProcessor(4096, 1, 1);
      procRef.current.onaudioprocess = (e) => {
        if (!streamingRef.current || !wsRef.current || wsRef.current.readyState !== 1) return;
        const d = e.inputBuffer.getChannelData(0);
        const pcm = new Int16Array(d.length);
        let rms = 0;
        for (let i = 0; i < d.length; i++) {
          pcm[i] = Math.max(-32768, Math.min(32767, d[i] * 32767));
          rms += d[i] * d[i];
        }
        waveBuf.current.push(Math.sqrt(rms / d.length) * 2.2);
        waveBuf.current.shift();
        wsRef.current.send(pcm.buffer);
      };
      srcRef.current.connect(procRef.current);
      procRef.current.connect(micRef.current.destination);
      streamingRef.current = true;
      setStreaming(true);
      trajRef.current = [];
      setTraj([]);
      addLog("live analysis started");
    } catch (err) {
      addLog("mic error: " + err.message, true);
    }
  }, [addLog]);

  const stopMic = useCallback(() => {
    streamingRef.current = false;
    setStreaming(false);
    if (procRef.current) { procRef.current.disconnect(); procRef.current = null; }
    if (srcRef.current) { srcRef.current.disconnect(); srcRef.current = null; }
    if (wsRef.current && wsRef.current.readyState === 1) wsRef.current.send("END");
    addLog("live analysis stopped");
  }, [addLog]);

  const toggleMic = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== 1) {
      connect();
      setTimeout(() => (streamingRef.current ? null : startMic()), 400);
      return;
    }
    streamingRef.current ? stopMic() : startMic();
  }, [connect, startMic, stopMic]);

  const onEnroll = useCallback(
    async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const sid = speakerId.trim();
      if (!sid) { addLog("enter speaker_id first", true); return; }
      const r = await fetch(`/v1/enroll?speaker_id=${encodeURIComponent(sid)}`, { method: "POST", body: await f.arrayBuffer() });
      addLog("enrolled: " + JSON.stringify(await r.json()));
      e.target.value = "";
    },
    [speakerId, addLog]
  );

  const onFile = useCallback(
    async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const sid = speakerId.trim();
      const tr = manualTr.trim();
      const r = await fetch(
        `/v1/analyze?${sid ? "speaker_id=" + encodeURIComponent(sid) + "&" : ""}context=${CTX}${tr ? "&transcript=" + encodeURIComponent(tr) : ""}`,
        { method: "POST", body: await f.arrayBuffer() }
      );
      const m = await r.json();
      if (m.verdict) showFinal(m.verdict);
      else addLog(JSON.stringify(m), true);
      e.target.value = "";
    },
    [speakerId, manualTr, showFinal, addLog]
  );

  // waveform draw loop
  useEffect(() => {
    const iv = setInterval(() => {
      const c = waveRef.current;
      if (!c) return;
      const g = c.getContext("2d");
      g.clearRect(0, 0, c.width, c.height);
      g.strokeStyle = "rgba(37,99,235,.15)";
      g.beginPath();
      g.moveTo(0, c.height / 2);
      g.lineTo(c.width, c.height / 2);
      g.stroke();
      const bw = c.width / waveBuf.current.length;
      waveBuf.current.forEach((v, i) => {
        const h = Math.max(2, Math.min(1, Math.abs(v)) * c.height);
        g.fillStyle = v >= 0 ? "rgba(37,99,235,.7)" : "rgba(37,99,235,.35)";
        g.fillRect(i * bw + 1, (c.height - h) / 2, bw - 2, h);
      });
    }, 60);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    fetch("/healthz")
      .then((r) => r.json())
      .then((h) =>
        setHealth(
          `layer A (artifact CNN): ${h.layers.artifact ? "ONLINE" : "offline"} · layer B (wav2vec2): ${h.layers.ssl ? "ONLINE" : "offline"} · enrolled: ${h.layers.speaker_enrollments}`
        )
      )
      .catch(() => setHealth("server offline — run: uvicorn swarakavach.server:app --port 8000"));
    fetchAudit();
  }, [fetchAudit]);

  const lvlClass = { LOW: "#059669", MID: "#b45309", HIGH: "#c2410c", CRITICAL: "#dc2626", STANDBY: "#94a3b8" }[level];

  return (
    <>
      <header className="hdr">
        <h1>
          Swara<span>Kavach</span> · Voice Integrity Console
        </h1>
        <span className="badge">SIH 2026 · PS26104 · AICTE</span>
        <div className="conn">
          <span className="dot" style={{ background: conn ? "#059669" : "#dc2626" }} />
          <span>{conn ? "STREAM READY" : "DISCONNECTED"}</span>
        </div>
      </header>

      <div className="main">
        <section className="card">
          <h2>Call Channel</h2>
          <div className="row">
            <button className={streaming ? "rec" : ""} onClick={toggleMic}>
              {streaming ? "Stop Live Analysis" : "Start Live Analysis"}
            </button>
          </div>
          <div className="row">
            <input type="text" value={speakerId} onChange={(e) => setSpeakerId(e.target.value)} placeholder="speaker_id (e.g. regional_manager)" />
          </div>
          <div className="row">
            <textarea rows={2} value={manualTr} onChange={(e) => setManualTr(e.target.value)} placeholder="call transcript (leave EMPTY for live ASR — whisper transcribes in real time; type to override)" />
          </div>
          <div className="row">
            <button
              onClick={() => {
                document.getElementById("enrollFile").click();
              }}
            >
              Enroll Voiceprint
            </button>
            <input type="file" id="enrollFile" accept="audio/*" hidden onChange={onEnroll} />
          </div>
          <div className="row">
            <button
              onClick={() => {
                document.getElementById("audioFile").click();
              }}
            >
              Analyze Audio File
            </button>
            <input type="file" id="audioFile" accept="audio/*" hidden onChange={onFile} />
          </div>
          <h2 className="mt">Live Signal</h2>
          <canvas ref={waveRef} className="full" width={298} height={64} style={{ marginTop: 0 }} />
          <h2 className="mt">Call Context</h2>
          <div className="ctxRow">
            <span className="k">caller-ID match</span>
            <span className="v">false</span>
          </div>
          <div className="ctxRow">
            <span className="k">transaction</span>
            <span className="v">₹50,00,000</span>
          </div>
          <div className="hint">Context feeds Detector D (metadata rules). Wire your CRM for real deployments.</div>
          <div className="status mono">{health}</div>
        </section>

        <section className="card">
          <h2>Impersonation Risk</h2>
          <div className="gaugeWrap">
            <div className="radar">
              <div className="sweep" />
              <svg className="rings" width="264" height="132" viewBox="0 0 264 132">
                <g fill="none" stroke="rgba(37,99,235,.12)">
                  <path d="M 22 132 A 110 110 0 0 1 242 132" />
                  <path d="M 52 132 A 80 80 0 0 1 212 132" />
                  <path d="M 82 132 A 50 50 0 0 1 182 132" />
                </g>
              </svg>
              <svg className="gauge" width="260" height="150" viewBox="0 0 260 150">
                <path d="M 130 135 L 30 135 A 100 100 0 0 1 59.3 64.4 L 130 135" fill="rgba(5,150,105,.07)" />
                <path d="M 130 135 L 59.3 64.4 A 100 100 0 0 1 114.4 36.2 L 130 135" fill="rgba(180,83,9,.06)" />
                <path d="M 130 135 L 114.4 36.2 A 100 100 0 0 1 175.4 45.9 L 130 135" fill="rgba(194,65,12,.06)" />
                <path d="M 130 135 L 175.4 45.9 A 100 100 0 0 1 230 135 L 130 135" fill="rgba(220,38,38,.07)" />
                <path d="M 30 135 A 100 100 0 0 1 230 135" fill="none" stroke="#e2e8f0" strokeWidth="12" strokeLinecap="round" />
                <path
                  d="M 30 135 A 100 100 0 0 1 230 135"
                  fill="none"
                  stroke={LVL_COLORS[level] || "#2563eb"}
                  strokeWidth="12"
                  strokeLinecap="round"
                  strokeDasharray={risk > 0 ? "314" : "0 999"}
                  strokeDashoffset={314 * (1 - risk)}
                  style={{ transition: "stroke-dashoffset .25s ease, stroke .25s" }}
                />
                <g stroke="#cbd5e1" strokeWidth="2">
                  <line x1="50.2" y1="87.2" x2="61.5" y2="92.5" />
                  <line x1="42.8" y1="52.6" x2="54.4" y2="55.7" />
                  <line x1="73.4" y1="24.8" x2="79.2" y2="35.6" />
                </g>
                <text x="130" y="110" textAnchor="middle" fill="#0f172a" fontSize="34" fontWeight="700" fontFamily="var(--font-mono)">
                  {risk > 0 || level !== "STANDBY" ? risk.toFixed(2) : "—"}
                </text>
                <text x="130" y="130" textAnchor="middle" fill="#94a3b8" fontSize="10" letterSpacing="2">
                  FUSED RISK
                </text>
              </svg>
            </div>
            <div className="level mono" style={{ color: lvlClass, letterSpacing: 6 }}>
              {level}
            </div>
            <div className="actions">
              {actions.map((a) => (
                <span key={a} className="action mono" style={{ color: a.includes("HOLD") ? "#dc2626" : a.includes("ALERT") || a.includes("WEBHOOK") ? "#c2410c" : a.includes("LIVENESS") ? "#b45309" : "#64748b", borderColor: "currentColor" }}>
                  {a.replace(/_/g, " ")}
                </span>
              ))}
            </div>
          </div>
          <canvas ref={sparkRef} className="full" width={640} height={120} />
          {finalV && (
            <div className="final">
              <b>FINAL VERDICT</b> — level <b style={{ color: LVL_COLORS[finalV.level] }}>{finalV.level}</b> · final {finalV.risk_final} · peak {finalV.risk_peak}
              <br />
              actions: <b>{finalV.actions.join(", ") || "none"}</b>
              <div className="bd">
                {Object.entries(finalV.breakdown || {})
                  .filter(([, v]) => v != null)
                  .map(([k, v]) => (
                    <span key={k}>
                      {k.replace(/_r?isk/g, "")}: <b>{v}</b> ·{" "}
                    </span>
                  ))}
              </div>
            </div>
          )}
        </section>

        <section className="card">
          <h2>Layer Contributions</h2>
          <div className="layers">
            {LAYERS.map(([k, q]) => {
              const v = layers[k];
              return (
                <div className="layer" key={k}>
                  <div className="t">
                    <span className="q">{q}</span>
                    <span>{v == null ? "ABSTAIN" : typeof v === "number" ? v.toFixed(2) : "—"}</span>
                  </div>
                  <div className="bar">
                    <i style={{ width: (v == null ? 0 : v * 100) + "%", backgroundColor: v == null ? "#cbd5e1" : riskColor(v) }} />
                    <div className="marks"><b /><b /><b /><b /></div>
                  </div>
                </div>
              );
            })}
          </div>
          <h2 className="mt">Live Transcript (ASR)</h2>
          <div className="status" style={{ fontStyle: "italic", minHeight: 34 }}>
            {asrText || "— auto-transcribes when the transcript box is empty —"}
          </div>
          <h2 className="mt">Incident Ledger (hash-chained)</h2>
          <div className="status">
            {audit ? (
              <>
                chain <b style={{ color: audit.chain_valid ? "#059669" : "#dc2626" }}>{audit.chain_valid ? "INTACT" : "BROKEN"}</b> · {audit.records.filter((r) => r.type === "incident").length} incidents
                {audit.records
                  .filter((r) => r.type === "incident")
                  .slice(-3)
                  .reverse()
                  .map((r) => (
                    <div key={r.incident_id} className="mono">
                      {r.incident_id} · {r.level} · risk {r.risk_peak} · {r.transaction_action}
                    </div>
                  ))}
              </>
            ) : (
              "no incidents yet"
            )}
          </div>
          <h2 className="mt">Event Log</h2>
          <div className="log">
            {log.map((l) => (
              <div key={l.id} className={"ev" + (l.warn ? " warn" : "")}>
                <span className="t">{l.time}</span> {l.s}
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="footer">
        5-layer detection: A spectral artifacts · B SSL wav2vec2 · C voiceprint · D call context · E intent · verdicts append to hash-chained incident ledger
      </div>
    </>
  );
}
