from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from bellu.config import load_config
from bellu.io.devices import Microphone, Speaker
from bellu.orchestrator import DuplexRuntime


def setup_sta(cfg: dict) -> None:
    """Clone Easy-Turn, download checkpoint + Qwen2.5-0.5B, write a patched train.yaml."""

    import shutil
    import subprocess

    import yaml

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


def build_runtime(cfg: dict, mock: bool) -> DuplexRuntime:
    if mock:
        from bellu.brain.llm import MockBrain
        from bellu.expression.tts import MockTTS
        from bellu.perception.asr import MockASR
        from bellu.perception.sta import MockSTA

        asr, sta, brain, tts = MockASR(), MockSTA(), MockBrain(), MockTTS()
    else:
        from bellu.brain.llm import SarvamBrain
        from bellu.expression.tts import ParlerExpression
        from bellu.perception.asr import IndicTranscribeASR
        from bellu.perception.sta import EasyTurnSTA

        asr = IndicTranscribeASR(cfg["asr"]["model_id"], cfg["asr"].get("language"), cfg["asr"].get("device", "cuda"))
        sta = EasyTurnSTA(cfg["sta"])
        brain = SarvamBrain(cfg["llm"])
        tts = ParlerExpression(cfg["tts"])
        print("Loading ASR (Indic-Transcribe-core)...")
        asr.load()
        print("Loading STA (Easy-Turn)...")
        sta.load()
        print("Loading LLM (Sarvam-30B)...")
        brain.load()
        print("Loading TTS (ParlerTTS)...")
        tts.load()

    speaker = Speaker(cfg["playback"]["output_sample_rate"])
    return DuplexRuntime(cfg, asr, sta, brain, tts, speaker, Microphone)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bellu full-duplex voice runtime")
    parser.add_argument("command", choices=["run", "setup-sta", "mock"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "setup-sta":
        setup_sta(cfg)
        return
    runtime = build_runtime(cfg, mock=args.command == "mock")
    print("Listening. Ctrl+C to stop.")
    runtime.start()
    runtime.join()


if __name__ == "__main__":
    main()
