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

function addBubble(role, text) {
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
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.0;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}

function playPcm16(arrayBuffer) {
  if (!audioCtx) return;
  const int16 = new Int16Array(arrayBuffer);
  if (!int16.length) return;
  const f32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i += 1) f32[i] = int16[i] / 32768;
  const buffer = audioCtx.createBuffer(1, f32.length, 16000);
  buffer.copyToChannel(f32, 0);
  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(audioCtx.destination);
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
  audioCtx = new AudioContext();
  media = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
  });
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
  source.connect(processor);
  processor.connect(audioCtx.destination);
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
    addBubble("sys", "Connected — keep talking. The model listens continuously.");
  };

  socket.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      playPcm16(event.data);
      return;
    }
    const data = JSON.parse(event.data);
    if (data.type === "state") {
      if (data.transcript && data.transcript !== lastUserText) {
        lastUserText = data.transcript;
        addBubble("user", data.transcript);
      }
      if (data.last_text && data.last_text !== lastBotText) {
        lastBotText = data.last_text;
        addBubble("bot", data.last_text);
        speakFallback(data.last_text);
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

addBubble("sys", "Click Connect, allow the mic, then speak continuously.");
