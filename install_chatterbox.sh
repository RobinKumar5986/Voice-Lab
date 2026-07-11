#!/usr/bin/env bash
# Chatterbox-Turbo TTS install - ISOLATED venv, separate from kokoro-env.
# Run this from inside the project folder.
#
# Why a separate venv: chatterbox-tts pulls its own torch/torchaudio
# versions which may conflict with the ones Kokoro needs.
set -e

echo "=== Creating virtual env (./chatterbox-env) ==="
python3 -m venv ./chatterbox-env
source ./chatterbox-env/bin/activate

echo "=== Installing PyTorch (CUDA build) ==="
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing chatterbox-tts + extras ==="
pip install -r requirements-chatterbox.txt

echo ""
echo "=== Done. Chatterbox is now installed alongside Kokoro. ==="
echo "The main app (character_voice_studio.py) will auto-detect it."