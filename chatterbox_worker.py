"""
chatterbox_worker.py - CLI bridge so character_voice_studio.py (running in
kokoro-env) can request a Chatterbox-Turbo generation without importing it
directly - Chatterbox lives in its own separate venv (chatterbox-env).

Called as a subprocess:
    ./chatterbox-env/bin/python3 chatterbox_worker.py \
        --text "..." --reference path/to/ref.wav --output path/to/out.wav \
        --exaggeration 0.5 --cfg-weight 0.5 --temperature 0.8 [--seed 12345]

Knobs (all affect *delivery*, never accent - accent comes from the reference):
  --exaggeration  emotional intensity / expressiveness (needs a lowish
                  --cfg-weight to really show).
  --cfg-weight    guidance / pacing. Lower = looser, more expressive; higher =
                  tighter, closer to a flat read.
  --temperature   sampling randomness. Higher = more variety between takes.
  --seed          fix the RNG for a reproducible take. -1 = random each run.

Unsupported knobs are dropped automatically based on the installed model's
generate() signature, so this stays compatible across Chatterbox versions.

Exits 0 on success. On failure, prints the error to stderr and exits 1.
"""

import argparse
import inspect
import os
import random
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS


def set_seed(seed: int):
    """Seed every RNG Chatterbox might touch, so output is reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass


def load_model():
    if torch.cuda.is_available():
        try:
            return ChatterboxTurboTTS.from_pretrained(device="cuda")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return ChatterboxTurboTTS.from_pretrained(device="cpu")


def supported_kwargs(func, candidate: dict) -> dict:
    """Keep only the kwargs this model's generate() actually accepts, so we
    don't crash on a Chatterbox build that lacks cfg_weight/temperature."""
    try:
        params = inspect.signature(func).parameters
    except (ValueError, TypeError):
        return dict(candidate)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(candidate)
    return {k: v for k, v in candidate.items() if k in params}


def generate(model, text, reference, kwargs):
    passable = supported_kwargs(model.generate, kwargs)
    dropped = [k for k in kwargs if k not in passable]
    if dropped:
        print(
            f"Chatterbox worker: this model ignores {', '.join(dropped)}.",
            file=sys.stderr,
        )
    return model.generate(text, audio_prompt_path=reference, **passable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exaggeration", type=float, default=0.5,
                        help="Emotional intensity / expressiveness (0.1-1.5).")
    parser.add_argument("--cfg-weight", dest="cfg_weight", type=float, default=0.5,
                        help="Guidance / pacing. Lower = looser & more expressive.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling randomness. Higher = more variety.")
    parser.add_argument("--seed", type=int, default=-1,
                        help="RNG seed for reproducible output. -1 = random.")
    args = parser.parse_args()

    # Seed BEFORE loading/generating so the whole run is deterministic.
    if args.seed is not None and args.seed >= 0:
        set_seed(args.seed)

    kwargs = {
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
        "temperature": args.temperature,
    }

    try:
        model = load_model()
        try:
            wav = generate(model, args.text, args.reference, kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(
                "Chatterbox worker: CUDA OOM during generation, retrying on CPU...",
                file=sys.stderr,
            )
            # Re-seed so the CPU retry reproduces the same intended take.
            if args.seed is not None and args.seed >= 0:
                set_seed(args.seed)
            model = ChatterboxTurboTTS.from_pretrained(device="cpu")
            wav = generate(model, args.text, args.reference, kwargs)
        ta.save(args.output, wav, model.sr)
    except Exception as e:
        print(f"Chatterbox worker error: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()