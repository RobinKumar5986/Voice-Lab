#!/bin/bash

# ============================================
# How to run this script:
#   chmod +x RC.sh
#   ./RC.sh
# ============================================

echo "Running character_voice_studio.py"
source ./kokoro-env/bin/activate
pip install -r requirements.txt
python3 character_voice_studio.py