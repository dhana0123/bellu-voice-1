const logEl = document.getElementById("log");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const statusEl = document.getElementById("status");
const micBtn = document.getElementById("mic");

function addBubble(role, text) {
  const el = document.createElement("div");
  el.className = `bubble ${role}`;
  el.textContent = text;
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
}

async function sendText(text) {
  const trimmed = text.trim();
  if (!trimmed) return;
  addBubble("user", trimmed);
  input.value = "";
  statusEl.textContent = "thinking";
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: trimmed }),
    });
    const data = await res.json();
    addBubble("bot", data.reply || "(empty reply)");
    statusEl.textContent = "connected";
    statusEl.className = "status ok";
  } catch (err) {
    addBubble("bot", "Could not reach the model.");
    statusEl.textContent = "offline";
    statusEl.className = "status bad";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendText(input.value);
});

async function ping() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const model = (data.model || "live").split("/").pop();
    statusEl.textContent = data.mode === "live" ? `connected · ${model}` : "connected · mock";
    statusEl.className = "status ok";
  } catch {
    statusEl.textContent = "offline";
    statusEl.className = "status bad";
  }
}

let recording = false;
let media = null;
let processor = null;
let chunks = [];
let ctx = null;

async function startMic() {
  ctx = new AudioContext();
  media = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = ctx.createMediaStreamSource(media);
  processor = ctx.createScriptProcessor(4096, 1, 1);
  chunks = [];
  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(ctx.destination);
  recording = true;
  micBtn.classList.add("on");
  statusEl.textContent = "listening";
}

function mergeChunks() {
  const length = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Float32Array(length);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

async function stopMic() {
  recording = false;
  micBtn.classList.remove("on");
  if (processor) processor.disconnect();
  if (media) media.getTracks().forEach((t) => t.stop());
  if (!chunks.length) return;
  const samples = mergeChunks();
  const sr = ctx ? ctx.sampleRate : 48000;
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = s < 0 ? s * 32768 : s * 32767;
  }
  statusEl.textContent = "transcribing";
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${proto}://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "transcript" && data.text) addBubble("user", data.text);
    if (data.type === "reply" && data.text) {
      addBubble("bot", data.text);
      statusEl.textContent = "connected";
      statusEl.className = "status ok";
    }
  };
  socket.onopen = () => {
    socket.send(JSON.stringify({ type: "hello", sr }));
    socket.send(pcm.buffer);
  };
}

micBtn.addEventListener("mousedown", () => startMic().catch(() => {
  statusEl.textContent = "mic blocked";
  statusEl.className = "status bad";
}));
micBtn.addEventListener("mouseup", () => { if (recording) stopMic(); });
micBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startMic(); });
micBtn.addEventListener("touchend", (e) => { e.preventDefault(); if (recording) stopMic(); });

ping();
addBubble("bot", "Say something, or type a message.");
