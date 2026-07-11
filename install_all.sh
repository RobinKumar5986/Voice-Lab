#!/usr/bin/env bash
# One-shot setup: installs both Kokoro and Chatterbox-Turbo environments.
set -e

echo "############################################"
echo "# Step 1/2: Installing Kokoro (core engine) #"
echo "############################################"
./install_kokoro.sh

echo ""
echo "################################################"
echo "# Step 2/2: Installing Chatterbox (emotion tags) #"
echo "################################################"
./install_chatterbox.sh

echo ""
echo "=== All done. Run the app with: ==="
echo "./RC.sh"