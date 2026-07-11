"""
chatterbox_worker.py - CLI bridge so character_voice_studio.py (running in
kokoro-env) can request a Chatterbox-Turbo generation without importing it
directly - Chatterbox lives in its own separate venv (chatterbox-env).

Called as a subprocess:
    ./chatterbox-env/bin/python3 chatterbox_worker.py \
        --text "..." --reference path/to/ref.wav --output path/to/out.wav

Exits 0 on success. On failure, prints the error to stderr and exits 1.

KNOWN LIMITATION: this reloads the Chatterbox model fresh on every call,
which adds a few seconds of overhead each time. Fine for occasional
emotional lines; if this becomes your main workflow, it's worth revisiting
as a persistent background process instead.
"""

import argparse
import os
import sys

# Must be set before torch is imported to take effect. Helps avoid CUDA OOM
# errors caused by memory fragmentation on small-VRAM GPUs.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS


def load_model():
    if torch.cuda.is_available():
        try:
            return ChatterboxTurboTTS.from_pretrained(device="cuda")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return ChatterboxTurboTTS.from_pretrained(device="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        model = load_model()
        try:
            wav = model.generate(args.text, audio_prompt_path=args.reference)
        except torch.cuda.OutOfMemoryError:
            # Ran out of VRAM mid-generation - retry once on CPU.
            torch.cuda.empty_cache()
            print(
                "Chatterbox worker: CUDA OOM during generation, retrying on CPU...",
                file=sys.stderr,
            )
            model = ChatterboxTurboTTS.from_pretrained(device="cpu")
            wav = model.generate(args.text, audio_prompt_path=args.reference)
        ta.save(args.output, wav, model.sr)
    except Exception as e:
        print(f"Chatterbox worker error: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()