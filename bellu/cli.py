from __future__ import annotations

import argparse
from pathlib import Path

from bellu.config import load_config
from bellu.orchestrator import DuplexRuntime


def setup_sta(cfg: dict) -> None:
    """Clone Easy-Turn, download checkpoint + Qwen2.5-0.5B, write a patched train.yaml."""

    import shutil
    import subprocess

    import yaml
    from huggingface_hub import snapshot_download

    root = Path("third_party/Easy-Turn")
    if not root.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/ASLP-lab/Easy-Turn.git", str(root)])

    models = Path("models/easy-turn")
    models.mkdir(parents=True, exist_ok=True)
    ckpt_dir = snapshot_download(cfg["sta"]["repo_id"], local_dir=str(models))
    src_yaml = root / "Easy_Turn" / "examples" / "wenetspeech" / "whisper" / "conf" / "train.yaml"
    qwen = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir="models/Qwen2.5-0.5B-Instruct")
    with src_yaml.open("r", encoding="utf-8") as handle:
        train = yaml.safe_load(handle)
    train["llm_path"] = str(Path(qwen).resolve())
    train["tokenizer_conf"]["llm_path"] = str(Path(qwen).resolve())
    dest = Path(cfg["sta"]["train_yaml"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(train, handle)
    ckpt = Path(ckpt_dir) / "checkpoint.pt"
    if ckpt.exists() and Path(cfg["sta"]["checkpoint"]) != ckpt:
        shutil.copy2(ckpt, cfg["sta"]["checkpoint"])
    print("Easy-Turn ready.")
    print(f"  repo: {root}")
    print(f"  checkpoint: {cfg['sta']['checkpoint']}")
    print(f"  qwen: {qwen}")


def build_runtime(cfg: dict, mock: bool, for_ui: bool = False) -> DuplexRuntime:
    """Build duplex stack. UI mode uses browser mic/speaker (NullSpeaker, no local Microphone)."""

    if mock:
        from bellu.brain.mock import MockBrain
        from bellu.perception.mock_asr import MockASR
        from bellu.perception.mock_sta import MockSTA, MockTTS

        asr, sta, brain, tts = MockASR(), MockSTA(), MockBrain(), MockTTS()
    else:
        from bellu.brain.llm import SarvamBrain
        from bellu.perception.asr import IndicTranscribeASR
        from bellu.perception.mock_sta import MockSTA, MockTTS

        asr = IndicTranscribeASR(cfg["asr"]["model_id"], cfg["asr"].get("language"), cfg["asr"].get("device", "cuda"))
        brain = SarvamBrain(cfg["llm"])
        print("Loading ASR (Indic-Transcribe-core)...")
        asr.load()
        print(f"Loading LLM ({cfg['llm']['model_id']})...")
        brain.load()

        sta = MockSTA()
        if not for_ui:
            try:
                from bellu.perception.sta import EasyTurnSTA

                print("Loading STA (Easy-Turn)...")
                sta = EasyTurnSTA(cfg["sta"])
                sta.load()
            except Exception as exc:
                print(f"Easy-Turn unavailable ({exc}); using MockSTA")

        tts = MockTTS()
        try:
            from bellu.expression.tts import ParlerExpression

            print("Loading TTS (ParlerTTS)...")
            tts = ParlerExpression(cfg["tts"])
            tts.load()
        except Exception as exc:
            print(f"ParlerTTS unavailable ({exc}); text-only voice replies")

    from bellu.io.devices import NullSpeaker

    if for_ui:
        speaker = NullSpeaker(cfg["playback"]["output_sample_rate"])
        return DuplexRuntime(cfg, asr, sta, brain, tts, speaker, None)

    from bellu.io.devices import Microphone, Speaker

    speaker = Speaker(cfg["playback"]["output_sample_rate"])
    return DuplexRuntime(cfg, asr, sta, brain, tts, speaker, Microphone)


def serve(cfg: dict, mock: bool, host: str, port: int) -> None:
    import uvicorn

    from bellu.ui.server import create_app

    runtime = build_runtime(cfg, mock=mock, for_ui=True)
    app = create_app(
        runtime,
        mode="mock" if mock else "live",
        model_id=cfg.get("llm", {}).get("model_id"),
    )
    print(f"Duplex UI: http://127.0.0.1:{port}")
    print("Open the page, click Connect, then talk continuously (Moshi-style).")
    uvicorn.run(app, host=host, port=port, log_level="info", ws="websockets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bellu full-duplex voice runtime")
    parser.add_argument("command", choices=["run", "setup-sta", "mock", "serve"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--mock", action="store_true", help="use mock models with serve")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "setup-sta":
        setup_sta(cfg)
        return
    if args.command == "serve":
        ui = cfg.get("ui") or {}
        serve(
            cfg,
            mock=args.mock,
            host=args.host or ui.get("host") or "0.0.0.0",
            port=args.port or int(ui.get("port") or 8998),
        )
        return
    runtime = build_runtime(cfg, mock=args.command == "mock", for_ui=False)
    print("Listening. Ctrl+C to stop.")
    runtime.start()
    runtime.join()


if __name__ == "__main__":
    main()
