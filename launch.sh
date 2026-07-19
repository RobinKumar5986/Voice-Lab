#!/usr/bin/env bash
# Runs RC.sh with no visible terminal window -- output (including the
# pip install step and any errors) goes to launch.log in this same
# folder instead, so nothing's lost, it's just not popping up on screen.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

LOG="$DIR/launch.log"
./RC.sh > "$LOG" 2>&1
STATUS=$?

if [ $STATUS -ne 0 ] && command -v notify-send >/dev/null 2>&1; then
    notify-send "CharacterVoiceStudio failed to start" "See launch.log in the project folder for details."
fi

exit $STATUS
