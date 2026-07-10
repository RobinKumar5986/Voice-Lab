#!/usr/bin/env bash
# Kokoro-82M TTS install for Ubuntu 24.04 (GTX 1650)
# Run this from inside the project folder (e.g. ~/Desktop/Text to speech)
set -e

echo "=== Installing system dependency: espeak-ng ==="
sudo apt-get update
sudo apt-get install -y espeak-ng

echo "=== Creating virtual env (./kokoro-env) ==="
python3 -m venv ./kokoro-env
source ./kokoro-env/bin/activate

echo "=== Installing PyTorch (CUDA 12.1 build, works with GTX 1650 drivers) ==="
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing Kokoro + deps ==="
pip install "kokoro>=0.9.4" soundfile misaki[en]

echo ""
echo "=== Done. To use it: ==="
echo "source ./kokoro-env/bin/activate"
echo "python3 character_voice_studio.py"