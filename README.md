# 🎙️ Voice Lab

**A local, offline character voice studio — mix voices, clone voices, and direct full emotional conversations, all on your own machine.**

Built on [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) for fast narration and [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) for real, trained paralinguistic emotion (laughs, sighs, coughs, and more). No cloud, no API keys, no subscriptions — everything runs locally.

<p align="center">
  <img width="2086" height="1444" alt="Voice Lab main window" src="https://github.com/user-attachments/assets/a74aeb95-2da3-45a8-8c01-0e99a5539bd8" />
</p>

<p align="center">
  <img width="49%" alt="Create New Voice - Mix" src="https://github.com/user-attachments/assets/a2fb6da8-eb38-4ee1-8212-757a6306d197" />
  <img width="49%" alt="Create New Voice - Clone" src="https://github.com/user-attachments/assets/1296f9f3-6e82-496a-98d0-9f9921ccf68e" />
</p>

---

## ✨ Features

- **54 bundled voices across 9 languages** (English US/UK, Spanish, French, Hindi, Italian, Japanese, Portuguese, Mandarin) via Kokoro-82M
- **🎛️ Voice Mixing** — blend any two bundled voices with an adjustable balance slider to create a brand-new deterministic voice
- **🧬 Voice Cloning** — import any clean recording (5+ seconds) and clone it with Chatterbox-Turbo, with full control over exaggeration, pacing, temperature, and seed for reproducible takes
- **🎭 Multi-character dialogue** — tag lines with `[Name]` and the app automatically switches speakers mid-generation, no manual splitting required
- **😄 Real emotion tags** — `[laugh]`, `[sigh]`, `[cough]`, `[chuckle]`, `[gasp]`, `[groan]`, `[sniff]`, `[shush]`, `[clear throat]` trigger genuine trained paralinguistic performance, not just narrated words
- **⏱️ Precise pause control** — `{pause:N}` inserts exact, controllable silence instead of hoping punctuation does the job
- **🎚️ Preview "Takes" system** — every generated preview is logged and replayable, so you can audition multiple takes and pin the exact one you like to a saved voice
- **⭐ My Voices library** — save, preview, and delete your custom mixed/cloned voices, all persisted locally
- **🖥️ Built for modest hardware** — tested and tuned on a GTX 1650 (4GB VRAM); Kokoro runs on CPU so the GPU stays free for Chatterbox

---

## 🔊 Hear it in action

| Sample | What it demonstrates |
|---|---|
| [🔊 am_eric.wav](https://github.com/user-attachments/files/29946235/am_eric.wav) | A bundled male voice (Eric) |
| [🔊 af_aoede.wav](https://github.com/user-attachments/files/29946237/af_aoede.wav) | A bundled female voice (Aoede) |
| [🔊 mix_bella_x_Nicole.wav](https://github.com/user-attachments/files/29946233/mix_bella_x_Nicole.wav) | A **mixed** voice — Bella blended with Nicole |
| [🔊 original_modi_ji_sound.mp3](https://github.com/user-attachments/files/29946231/original_modi_ji_sound.mp3) → [🔊 modi_ji.wav](https://github.com/user-attachments/files/29946232/modi_ji.wav) | A **cloned** voice — the second file is the clone generated from the first reference clip |
| [🔊 Eric_Nicole_Talk.wav](https://github.com/user-attachments/files/29946234/Eric_Nicole_Talk.wav) | A full **multi-character conversation with emotion tags** |

*(Click any filename above to download and listen — GitHub doesn't allow inline audio embeds from outside its own upload flow.)*

The multi-character sample was generated from this exact prompt, with no editing after generation:

```
[Nicole] Hey, did you actually finish the report?
[Eric] [laugh] Barely. I sent it like two minutes ago.
[Nicole] Of course you did. Cutting it close as always.
[Eric] [sigh] Hey, it got done, didn't it?
[Nicole] [chuckle] Fair enough. Buy me a coffee and we're even.
[Eric] Deal.
```

That's the whole workflow in one clip: two distinct speakers auto-detected from `[Name]` tags, real laughs and sighs, and natural pacing — straight out of the app.

---

## 🚀 Getting Started

### Requirements

- Ubuntu (tested on 24.04) with Python 3
- `espeak-ng` system package
- An NVIDIA GPU is recommended for Chatterbox but not required — everything falls back gracefully to CPU

### One-shot setup

```bash
git clone https://github.com/RobinKumar5986/Voice-Lab.git
cd Voice-Lab
chmod +x setup_all.sh
./setup_all.sh
```

This installs both environments:
- `kokoro-env` — the core narration engine (required)
- `chatterbox-env` — emotion tags and voice cloning (optional, isolated in its own venv so it can never break Kokoro)

### Run it

```bash
chmod +x RC.sh
./RC.sh
```

> Don't want emotion tags or cloning? Skip `install_chatterbox.sh` entirely — the app detects it's missing and runs in fast, Kokoro-only mode automatically.

---

## 🧠 How it works

Voice Lab routes each line of your script to whichever engine fits it best:

- Plain narration → **Kokoro-82M** (fast, deterministic, runs on CPU)
- A line containing an enabled emotion tag → **Chatterbox-Turbo** (slower, stochastic, GPU-accelerated when available)
- A `[Name]` tag → switches the active speaker for everything that follows, either from an explicit assignment in **🎭 Characters** or auto-detected against any existing voice name
- A `{pause:N}` marker → real silence spliced in at that exact point

Custom voices come in two flavors:

- **Mix** — a weighted blend of two bundled Kokoro voices. Fully deterministic: the same blend + text always sounds identical.
- **Clone** — built from an imported recording via Chatterbox. Accent and timbre come entirely from the reference clip; sliders only shape delivery (exaggeration, pacing, variety). Every preview is a "take" you can audition and pin — the seed of the take you pick gets saved with the voice so it stops drifting between runs.

---

## 📜 License

Released under the **MIT License** — see [`LICENSE`](LICENSE) for the full text. Use it, fork it, ship it in your own projects.

---

## ☕ Support this project

Voice Lab is free, open-source, and built solo in my spare time. If it saved you time or you just want to see more tools like it, consider sponsoring or buying me a coffee — it genuinely helps keep projects like this going.

**⭐ Star the repo** — costs nothing and helps others find it.
**💖 Sponsor** — via the GitHub Sponsors button at the top of this repo, if enabled.
**☕ Donate** — every bit is appreciated and goes straight back into building more of this.

Thanks for checking out Voice Lab! Issues, feature requests, and pull requests are all welcome.
