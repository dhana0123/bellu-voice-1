const logEl = document.getElementById("log");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const statusEl = document.getElementById("status");
const connectBtn = document.getElementById("connect");
const interruptBtn = document.getElementById("interrupt");
const meterEl = document.getElementById("meter");
const turnEl = document.getElementById("turn");

let socket = null;
let audioCtx = null;
let media = null;
let processor = null;
let playTime = 0;
let lastUserText = "";
let lastBotText = "";
let connected = false;

function displayable(text) {
  if (!text) return "";
  const trimmed = String(text).trim();
  if (trimmed.length > 180) return "";
  if (/TRIGGER:|CURRENT SNAPSHOT|ASR transcript/i.test(trimmed)) return "";
  const parts = trimmed.split(/\s+/);
  if (parts.length >= 6 && new Set(parts).size <= 2) return "";
  return trimmed;
}
  if (!text) return;
  const el = document.createElement("div");
  el.className = `bubble ${role}`;
  el.textContent = text;
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = `status ${cls || ""}`;
}

function speakFallback(text) {
  if (!text || !window.speechSynthesis) return;
  if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.0;
  utter.volume = 1.0;
  const voices = window.speechSynthesis.getVoices();
  const hi = voices.find((v) => /hi|india|hindi/i.test(`${v.lang} ${v.name}`));
  if (hi) utter.voice = hi;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}

function playPcm16(arrayBuffer) {
  if (!audioCtx) return;
  if (audioCtx.state === "suspended") audioCtx.resume();
  const int16 = new Int16Array(arrayBuffer);
  if (!int16.length) return;
  const srcRate = 16000;
  const dstRate = audioCtx.sampleRate || srcRate;
  const f32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i += 1) f32[i] = int16[i] / 32768;
  let samples = f32;
  if (Math.abs(dstRate - srcRate) > 1) {
    const n = Math.max(1, Math.round(f32.length * dstRate / srcRate));
    const out = new Float32Array(n);
    const scale = (f32.length - 1) / Math.max(n - 1, 1);
    for (let i = 0; i < n; i += 1) {
      const x = i * scale;
      const i0 = Math.floor(x);
      const i1 = Math.min(i0 + 1, f32.length - 1);
      const t = x - i0;
      out[i] = f32[i0] * (1 - t) + f32[i1] * t;
    }
    samples = out;
  }
  const buffer = audioCtx.createBuffer(1, samples.length, dstRate);
  buffer.copyToChannel(samples, 0);
  const src = audioCtx.createBufferSource();
  const gain = audioCtx.createGain();
  gain.gain.value = 1.0;
  src.buffer = buffer;
  src.connect(gain);
  gain.connect(audioCtx.destination);
  const now = audioCtx.currentTime;
  if (playTime < now) playTime = now + 0.02;
  src.start(playTime);
  playTime += buffer.duration;
}

function floatTo16(input) {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out;
}

async function startMicStream() {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  audioCtx = new Ctx();
  await audioCtx.resume();
  playTime = audioCtx.currentTime;
  const blip = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  blip.frequency.value = 660;
  gain.gain.value = 0.08;
  blip.connect(gain);
  gain.connect(audioCtx.destination);
  blip.start();
  blip.stop(audioCtx.currentTime + 0.12);

  media = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
  });
  await audioCtx.resume();
  const source = audioCtx.createMediaStreamSource(media);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const inputData = event.inputBuffer.getChannelData(0);
    const pcm = floatTo16(inputData);
    socket.send(pcm.buffer);
    let sum = 0;
    for (let i = 0; i < inputData.length; i += 1) sum += inputData[i] * inputData[i];
    const rms = Math.sqrt(sum / inputData.length);
    meterEl.textContent = `mic · ${(rms * 100).toFixed(1)}`;
  };
  const mute = audioCtx.createGain();
  mute.gain.value = 0;
  source.connect(processor);
  processor.connect(mute);
  mute.connect(audioCtx.destination);
}

function stopMicStream() {
  if (processor) processor.disconnect();
  if (media) media.getTracks().forEach((t) => t.stop());
  processor = null;
  media = null;
}

async function connect() {
  if (connected) {
    disconnect();
    return;
  }
  setStatus("connecting", "live");
  await startMicStream();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/api/chat`);
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    connected = true;
    connectBtn.classList.add("on");
    connectBtn.textContent = "Live";
    setStatus("live duplex", "ok");
    socket.send(JSON.stringify({ type: "hello", sr: audioCtx.sampleRate }));
    addBubble("sys", "Connected — Telugu only. Keep talking.");
  };

  socket.onmessage = async (event) => {
    let binary = null;
    if (event.data instanceof ArrayBuffer) binary = event.data;
    else if (event.data instanceof Blob) binary = await event.data.arrayBuffer();
    if (binary) {
      playPcm16(binary);
      return;
    }
    const data = JSON.parse(event.data);
    if (data.type === "state") {
      if (data.transcript && data.transcript !== lastUserText) {
        lastUserText = data.transcript;
        const shown = displayable(data.transcript);
        if (shown) addBubble("user", shown);
      }
      if (data.last_text && data.last_text !== lastBotText) {
        lastBotText = data.last_text;
        const shown = displayable(data.last_text);
        if (shown) {
          addBubble("bot", shown);
          speakFallback(shown);
        }
      }
      turnEl.textContent = `turn · ${data.turn || "—"} · ${data.assistant || "waiting"}`;
      if (typeof data.rms === "number") meterEl.textContent = `mic · ${(data.rms * 100).toFixed(1)}`;
    } else if (data.type === "reply" && data.text) {
      addBubble("bot", data.text);
      speakFallback(data.text);
    } else if (data.type === "log" && data.kind === "protocol") {
      // already covered by last_text usually
    } else if (data.type === "hello") {
      setStatus(`live · ${(data.model || "").split("/").pop()}`, "ok");
    }
  };

  socket.onclose = () => disconnect(false);
  socket.onerror = () => setStatus("socket error", "bad");
}

function disconnect(closeSocket = true) {
  connected = false;
  connectBtn.classList.remove("on");
  connectBtn.textContent = "Connect";
  stopMicStream();
  if (closeSocket && socket) socket.close();
  socket = null;
  setStatus("offline", "bad");
  meterEl.textContent = "mic · idle";
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addBubble("user", text);
  input.value = "";
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "chat", text }));
  } else {
    fetch("/api/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    })
      .then((r) => r.json())
      .then((data) => {
        addBubble("bot", data.reply || "");
        speakFallback(data.reply || "");
      });
  }
});

interruptBtn.addEventListener("click", () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "interrupt" }));
  }
  if (window.speechSynthesis) window.speechSynthesis.cancel();
});

connectBtn.addEventListener("click", () => {
  connect().catch((err) => {
    console.error(err);
    setStatus("mic blocked", "bad");
    disconnect();
  });
});

fetch("/api/health")
  .then((r) => r.json())
  .then((data) => {
    const model = (data.model || "live").split("/").pop();
    setStatus(data.mode === "live" ? `ready · ${model}` : "ready · mock", "ok");
  })
  .catch(() => setStatus("offline", "bad"));

addBubble("sys", "Click Connect, allow the mic, then speak Telugu.");
