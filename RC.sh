#!/bin/bash

# ============================================
# How to run this script:
#   chmod +x RC.sh
#   ./RC.sh
#
# NOTE: Emotion tags ([laugh], [cough], etc.) require chatterbox-env to
# also be installed (run install_chatterbox.sh once, separately). Without
# it, the app still works fine - it just falls back to Kokoro-only.
# ============================================

echo "Running character_voice_studio.py"
source ./kokoro-env/bin/activate
# pip install -r requirements.txt
python3 character_voice_studio.py