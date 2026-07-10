"""
Quick test of Kokoro-82M TTS.
Run: python3 test_kokoro.py
Produces sample_0.wav, sample_1.wav, etc. in the current directory.
"""
import torch
import soundfile as sf
from kokoro import KPipeline

# lang_code='a' = American English. Use 'b' for British English.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pipeline = KPipeline(lang_code="a", device=device)

text = """
This is a test of Kokoro text to speech, running locally on Robin's machine.
If this sounds natural, the install worked.
"""

# Try a few voices so Robin can pick a favorite.
# Full voice list: https://github.com/hexgrad/kokoro/blob/main/kokoro.js/voices/README.md
voices_to_try = ["af_heart", "af_bella", "am_adam"]

for voice in voices_to_try:
    generator = pipeline(text, voice=voice, speed=1.0, split_pattern=r"\n+")
    for i, (graphemes, phonemes, audio) in enumerate(generator):
        filename = f"sample_{voice}_{i}.wav"
        sf.write(filename, audio, 24000)
        print(f"Wrote {filename}")

print("\nDone. Listen to the sample_*.wav files and pick your favorite voice.")