"""
Character Voice Studio - Kokoro TTS desktop UI (+ optional Chatterbox emotion tags)

Features:
1. Character (voice) selector with preview playback.
   - If no preview clip exists yet for a character, generates a 3-5s
     sample and saves it to ./characterSpeech/<voice>.wav
   - If it already exists, just plays it back.
2. Large text box to paste text for generation.
3. Generate button that renders the pasted text using the selected
   voice and saves the output to ./output/<timestamp>_<voice>.wav
4. Emotion tag toggles ([laugh], [sigh], [cough], etc.) - ON by default.
   If your text contains an enabled tag, that line is routed to
   Chatterbox-Turbo (real trained emotion) instead of Kokoro. Disabled
   tags are silently stripped from the text before generating.
5. Emotion intensity slider - controls how strongly Chatterbox leans into
   the emotional/expressive character of a tagged line.
6. {pause:N} marker for exact, controllable silence (e.g. {pause:0.6} for
   600ms) - real dead air inserted at that exact point, not something the
   TTS engine has to guess from punctuation.
7. Help button explaining all of the above with example prompts.
8. Custom voices ("My Voices") can be deleted via a trash icon next to
   each row, with a confirm dialog before anything is removed.

Run:
    source ./kokoro-env/bin/activate
    python3 character_voice_studio.py
"""

import os
import re
import subprocess
import tempfile
import threading
from datetime import datetime

import numpy as np

import customtkinter as ctk
import torch
import soundfile as sf
import pygame
from kokoro import KPipeline


from scipy.signal import resample_poly
from math import gcd

from voice_lab import VoiceRegistry, VoiceLabWindow, CUSTOM_PREVIEW_DIR


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHARACTER_SPEECH_DIR = "characterSpeech"
OUTPUT_DIR = "output"
LAST_TEXT_FILE = "last_text.txt"
SAMPLE_RATE = 24000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(CHARACTER_SPEECH_DIR, "references")
CHATTERBOX_ENV_PYTHON = os.path.join(SCRIPT_DIR, "chatterbox-env", "bin", "python3")
CHATTERBOX_WORKER_SCRIPT = os.path.join(SCRIPT_DIR, "chatterbox_worker.py")

# Official Chatterbox-Turbo paralinguistic tags (confirmed by Resemble AI staff).
# Note: these are documented as inconsistent/hit-or-miss, not a bug here.
EMOTION_TAGS = [
    "laugh", "chuckle", "cough", "sigh",
    "clear throat", "shush", "groan", "sniff", "gasp",
]

# "{pause:N}" inserts N seconds of real silence at that exact spot - see
# MARKER_RE / split_into_chunks below.
#
# {curly braces} are used rather than the [square brackets] the emotion tags
# use, on purpose: reusing [square brackets] would make strip_disabled_tags()
# treat "{pause:0.6}" as an unrecognized emotion tag and silently delete it,
# since it wouldn't be in the enabled-tags set.
MAX_PAUSE_SECONDS = 10.0  # sanity clamp so a typo can't stall generation for minutes
MARKER_RE = re.compile(
    r"\{\s*pause\s*:\s*([\d.]+)\s*\}",
    re.IGNORECASE,
)

# Longer sample text used ONLY to build a Chatterbox reference clip per voice
# (Chatterbox requires 5+ seconds of reference audio; the short previews
# below are too short for that purpose).
REFERENCE_TEXT = (
    "This is a longer voice sample being generated so it can be used as a "
    "reference clip for emotional voice cloning. It needs to run for about "
    "fifteen seconds so there is enough audio for the cloning model to work with."
)

# Preview line spoken in each language so the sample is meaningful,
# not just English text run through the wrong phonemizer.
PREVIEW_TEXTS = {
    "a": "Hi there! This is a quick preview of how I sound.",
    "b": "Hello there! This is a quick preview of how I sound.",
    "e": "Hola, esta es una vista previa rapida de mi voz.",
    "f": "Bonjour, ceci est un apercu rapide de ma voix.",
    "h": "Namaste, yeh meri awaaz ka ek preview hai.",
    "i": "Ciao, questa e una rapida anteprima della mia voce.",
    "j": "Kon'nichiwa, kore wa watashi no koe no puryu desu.",
    "p": "Ola, esta e uma previa rapida da minha voz.",
    "z": "Ni hao, zhe shi wo shengyin de kuaisu yulan.",
}

# (label shown in the language selector, lang_code, extra note)
LANGUAGES = [
    ("English (US)", "a", None),
    ("English (UK)", "b", None),
    ("Spanish", "e", None),
    ("French", "f", None),
    ("Hindi", "h", None),
    ("Italian", "i", None),
    ("Japanese", "j", "needs: pip install misaki[ja]"),
    ("Portuguese (Brazil)", "p", None),
    ("Mandarin Chinese", "z", "needs: pip install misaki[zh]"),
]

# (display name, voice id, lang_code) - all 54 bundled Kokoro voices
VOICES = [
    # American English
    ("Heart (Female)", "af_heart", "a"),
    ("Alloy (Female)", "af_alloy", "a"),
    ("Aoede (Female)", "af_aoede", "a"),
    ("Bella (Female)", "af_bella", "a"),
    ("Jessica (Female)", "af_jessica", "a"),
    ("Kore (Female)", "af_kore", "a"),
    ("Nicole (Female)", "af_nicole", "a"),
    ("Nova (Female)", "af_nova", "a"),
    ("River (Female)", "af_river", "a"),
    ("Sarah (Female)", "af_sarah", "a"),
    ("Sky (Female)", "af_sky", "a"),
    ("Adam (Male)", "am_adam", "a"),
    ("Echo (Male)", "am_echo", "a"),
    ("Eric (Male)", "am_eric", "a"),
    ("Fenrir (Male)", "am_fenrir", "a"),
    ("Liam (Male)", "am_liam", "a"),
    ("Michael (Male)", "am_michael", "a"),
    ("Onyx (Male)", "am_onyx", "a"),
    ("Puck (Male)", "am_puck", "a"),
    ("Santa (Male)", "am_santa", "a"),
    # British English
    ("Alice (Female)", "bf_alice", "b"),
    ("Emma (Female)", "bf_emma", "b"),
    ("Isabella (Female)", "bf_isabella", "b"),
    ("Lily (Female)", "bf_lily", "b"),
    ("Daniel (Male)", "bm_daniel", "b"),
    ("Fable (Male)", "bm_fable", "b"),
    ("George (Male)", "bm_george", "b"),
    ("Lewis (Male)", "bm_lewis", "b"),
    # Spanish
    ("Dora (Female)", "ef_dora", "e"),
    ("Alex (Male)", "em_alex", "e"),
    ("Santa (Male)", "em_santa", "e"),
    # French
    ("Siwis (Female)", "ff_siwis", "f"),
    # Hindi
    ("Alpha (Female)", "hf_alpha", "h"),
    ("Beta (Female)", "hf_beta", "h"),
    ("Omega (Male)", "hm_omega", "h"),
    ("Psi (Male)", "hm_psi", "h"),
    # Italian
    ("Sara (Female)", "if_sara", "i"),
    ("Nicola (Male)", "im_nicola", "i"),
    # Japanese
    ("Alpha (Female)", "jf_alpha", "j"),
    ("Gongitsune (Female)", "jf_gongitsune", "j"),
    ("Nezumi (Female)", "jf_nezumi", "j"),
    ("Tebukuro (Female)", "jf_tebukuro", "j"),
    ("Kumo (Male)", "jm_kumo", "j"),
    # Brazilian Portuguese
    ("Dora (Female)", "pf_dora", "p"),
    ("Alex (Male)", "pm_alex", "p"),
    ("Santa (Male)", "pm_santa", "p"),
    # Mandarin Chinese
    ("Xiaobei (Female)", "zf_xiaobei", "z"),
    ("Xiaoni (Female)", "zf_xiaoni", "z"),
    ("Xiaoxiao (Female)", "zf_xiaoxiao", "z"),
    ("Xiaoyi (Female)", "zf_xiaoyi", "z"),
    ("Yunjian (Male)", "zm_yunjian", "z"),
    ("Yunxi (Male)", "zm_yunxi", "z"),
    ("Yunxia (Male)", "zm_yunxia", "z"),
    ("Yunyang (Male)", "zm_yunyang", "z"),
]

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ---------------------------------------------------------------------------
# Emotion tag helpers
# ---------------------------------------------------------------------------

def strip_disabled_tags(text: str, enabled_tags: set) -> str:
    """Remove any [tag] from text whose tag name is NOT in enabled_tags.
    Enabled tags are left exactly as-is."""

    def _replace(match):
        tag_name = match.group(1).strip().lower()
        return match.group(0) if tag_name in enabled_tags else ""

    return re.sub(r"\[([^\[\]]+)\]", _replace, text)


def text_has_any_tag(text: str) -> bool:
    return bool(re.search(r"\[[^\[\]]+\]", text))


class VoiceEngine:
    """Wraps Kokoro pipelines, one per language code, created lazily."""

    def __init__(self):
        # Kokoro runs on CPU so the full GPU (only ~4GB on this machine)
        # stays free for Chatterbox, which needs it more.
        self.device = "cpu"
        self._pipelines = {}  # lang_code -> KPipeline
        self._lock = threading.Lock()

    def get_pipeline(self, lang_code: str) -> KPipeline:
        with self._lock:
            if lang_code not in self._pipelines:
                self._pipelines[lang_code] = KPipeline(
                    lang_code=lang_code, device=self.device
                )
            return self._pipelines[lang_code]

    def _build_blend_voice(self, pipeline, voice_id: str):
        """Build a single blended voice tensor from a spec like
        "af_bella(0.7),af_sky(0.3)" (or an unweighted "af_bella,af_sky").

        Base Kokoro only averages comma-separated voices *equally* and does
        NOT understand the (weight) syntax - it would try to load a voice
        literally named "af_bella(0.7)" and fail. So we resolve and weight the
        underlying voice tensors ourselves, then hand Kokoro the finished
        tensor through its voice cache.
        """
        loader = getattr(pipeline, "load_voice", None) or getattr(
            pipeline, "load_single_voice"
        )

        entries = []
        for part in voice_id.split(","):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^(.+?)\s*\(\s*([\d.]+)\s*\)\s*$", part)
            if m:
                entries.append((m.group(1).strip(), float(m.group(2))))
            else:
                entries.append((part, 1.0))
        if not entries:
            raise RuntimeError(f"Empty voice spec: {voice_id!r}")

        total = sum(w for _, w in entries) or 1.0
        blended = None
        for name, weight in entries:
            pack = loader(name)
            contrib = pack * (weight / total)
            blended = contrib if blended is None else blended + contrib
        return blended

    def synthesize(self, text: str, voice_id: str, lang_code: str, speed: float = 1.0):
        """Returns concatenated float32 numpy audio at SAMPLE_RATE.

        `voice_id` can be a single bundled voice ("af_bella") or a blend spec
        ("af_bella(0.7),af_sky(0.3)") from the "Create New Voice" panel
        (voice_lab.py) for mixed custom voices. Blends are resolved to a real
        voice tensor here (see _build_blend_voice) rather than passed to
        Kokoro as a string it cannot parse.
        """
        import numpy as np

        pipeline = self.get_pipeline(lang_code)

        voice_arg = voice_id
        if ("," in voice_id) or ("(" in voice_id):
            blended = self._build_blend_voice(pipeline, voice_id)
            # Stash under the spec string so Kokoro's load_voice returns it
            # directly (it checks its cache before loading from disk).
            try:
                pipeline.voices[voice_id] = blended
            except Exception:
                pass
            voice_arg = voice_id

        chunks = []
        generator = pipeline(text, voice=voice_arg, speed=speed, split_pattern=r"\n+")
        for _, _, audio in generator:
            chunks.append(audio)
        if not chunks:
            raise RuntimeError("No audio was generated for this text.")
        return np.concatenate(chunks)


class ChatterboxBridge:
    """Talks to Chatterbox-Turbo (installed in its own separate venv,
    chatterbox-env) via subprocess, so this app never has to import it
    directly and can't be broken by its dependency tree."""

    def __init__(self, kokoro_engine: VoiceEngine):
        self.kokoro_engine = kokoro_engine

    def _reference_clip_path(self, key: str) -> str:
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        return os.path.join(REFERENCE_DIR, f"{key}_reference.wav")

    def ensure_reference_clip(self, voice_id: str, lang_code: str) -> str:
        """Reference-clip builder for a single bundled voice (key == voice_id,
        since bundled voice ids are already filename-safe)."""
        return self.ensure_reference_clip_for_key(voice_id, voice_id, lang_code)

    def ensure_reference_clip_for_key(self, key: str, synth_voice_id: str, lang_code: str) -> str:
        """Same as ensure_reference_clip, but `key` (used only for the cache
        filename) can differ from `synth_voice_id` (what's actually handed to
        Kokoro). This lets a mixed voice like "af_bella(0.7),af_sky(0.3)" -
        not a valid filename - be cached under a clean custom-voice key
        instead."""
        ref_path = self._reference_clip_path(key)
        if os.path.exists(ref_path):
            return ref_path
        audio = self.kokoro_engine.synthesize(REFERENCE_TEXT, synth_voice_id, lang_code, speed=1.0)
        sf.write(ref_path, audio, SAMPLE_RATE)
        return ref_path

    def _run_worker(self, text: str, reference_clip: str, exaggeration: float,
                    seed=None, cfg_weight: float = 0.5, temperature: float = 0.8) -> np.ndarray:
        """`seed` (if not None) is forwarded to the worker so generation is
        reproducible - this is what pins a cloned voice's accent/take so it
        doesn't drift between runs. `cfg_weight` (guidance/pacing) and
        `temperature` (variety) are the other Chatterbox delivery knobs; the
        worker drops any the installed model doesn't support. Note: none of
        these change the ACCENT - accent comes from the reference clip."""
        if not os.path.exists(CHATTERBOX_ENV_PYTHON):
            raise RuntimeError(
                "Chatterbox isn't installed (chatterbox-env not found next to this "
                "script). Disable emotion tags to use fast Kokoro-only generation, "
                "or run install_chatterbox.sh first."
            )

        tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        cmd = [
            CHATTERBOX_ENV_PYTHON,
            CHATTERBOX_WORKER_SCRIPT,
            "--text", text,
            "--reference", reference_clip,
            "--output", tmp_path,
            "--exaggeration", str(exaggeration),
            "--cfg-weight", str(cfg_weight),
            "--temperature", str(temperature),
        ]
        if seed is not None:
            cmd += ["--seed", str(int(seed))]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Chatterbox generation failed:\n{result.stderr.strip()}")

            audio, sr = sf.read(tmp_path, dtype="float32")
            print(f"[DEBUG] Chatterbox output sr={sr}, len={len(audio)}")
            return resample_audio(audio, sr, SAMPLE_RATE)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def synthesize_chunk(
        self,
        text: str,
        voice_id: str,
        lang_code: str,
        exaggeration: float = 0.5,
        reference_key: str = None,
        seed=None,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> np.ndarray:
        """Generates ONE short chunk via Chatterbox and returns audio as a
        numpy array resampled to SAMPLE_RATE, ready to concatenate with
        Kokoro chunks. `exaggeration` controls how strongly the emotional/
        expressive character comes through (higher = more intense).
        `reference_key` overrides the cache filename - pass a custom voice's
        registry key here when `voice_id` itself isn't filename-safe (e.g. a
        mixed-voice blend string). `seed`/`cfg_weight`/`temperature` are the
        Chatterbox delivery knobs."""
        key = reference_key or voice_id
        reference_clip = self.ensure_reference_clip_for_key(key, voice_id, lang_code)
        return self._run_worker(text, reference_clip, exaggeration, seed=seed,
                                cfg_weight=cfg_weight, temperature=temperature)

    def synthesize_chunk_with_reference(
        self,
        text: str,
        reference_path: str,
        exaggeration: float = 0.5,
        seed=None,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> np.ndarray:
        """Generates ONE short chunk via Chatterbox using an explicit
        reference clip (e.g. a user-imported recording for a cloned custom
        voice) instead of one built from a Kokoro voice. `seed`/`cfg_weight`/
        `temperature` are the delivery knobs; accent comes from the clip."""
        return self._run_worker(text, reference_path, exaggeration, seed=seed,
                                cfg_weight=cfg_weight, temperature=temperature)


class CharacterVoiceStudio(ctk.CTk):
    def __init__(self):
        super().__init__()

        os.makedirs(CHARACTER_SPEECH_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.engine = VoiceEngine()
        self.chatterbox_bridge = ChatterboxBridge(self.engine)
        self.voice_registry = VoiceRegistry()
        self._save_after_id = None

        pygame.mixer.init()

        self.title("Character Voice Studio (Kokoro TTS)")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Adaptive sizing: use a large fraction of the screen, with a
        # generous floor so it never ends up cramped on any monitor.
        ctk.set_widget_scaling(1.15)
        ctk.set_window_scaling(1.15)

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(max(int(screen_w * 0.78), 1280), 1800)
        win_h = min(max(int(screen_h * 0.82), 860), 1200)
        pos_x = (screen_w - win_w) // 2
        pos_y = (screen_h - win_h) // 2
        self.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.minsize(1100, 760)

        # Scale factor so fonts/paddings adapt a bit to window size.
        # Floor raised so nothing ever renders too small.
        self.scale = max(1.15, min(1.6, win_w / 1150))

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _f(self, base_size: int) -> int:
        return int(base_size * self.scale)

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0, minsize=self._f(300))
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_character_panel()
        self._build_text_panel()

    # -- Left panel: character selection + preview -----------------------
    def _build_character_panel(self):
        panel = ctk.CTkFrame(self, corner_radius=12)
        panel.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)
        panel.grid_rowconfigure(4, weight=1)
        panel.grid_columnconfigure(0, weight=1)
        self.character_panel = panel

        title = ctk.CTkLabel(
            panel, text="Character", font=ctk.CTkFont(size=self._f(20), weight="bold")
        )
        title.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        # Language selector - filters which voices show up below. The final
        # entry is a pseudo-language that shows custom voices from the
        # "Create New Voice" panel instead of a bundled-language filter.
        language_labels = [lbl for lbl, _, _ in LANGUAGES] + ["★ My Voices"]
        self.language_menu = ctk.CTkOptionMenu(
            panel,
            values=language_labels,
            command=self._on_language_selected,
            font=ctk.CTkFont(size=self._f(14)),
            dropdown_font=ctk.CTkFont(size=self._f(14)),
            height=self._f(36),
        )
        self.language_menu.grid(row=1, column=0, padx=16, pady=(0, 4), sticky="ew")
        self.language_menu.set(language_labels[0])
        self.current_lang_code = LANGUAGES[0][1]

        self.language_note_label = ctk.CTkLabel(
            panel,
            text="",
            font=ctk.CTkFont(size=self._f(11)),
            text_color="orange",
            wraplength=self._f(260),
            justify="left",
        )
        self.language_note_label.grid(row=2, column=0, padx=16, pady=(0, 4), sticky="w")

        hint = ctk.CTkLabel(
            panel,
            text="Click a name to select it and hear a preview.",
            font=ctk.CTkFont(size=self._f(12)),
            text_color="gray60",
        )
        hint.grid(row=3, column=0, padx=16, pady=(0, 8), sticky="w")

        # Scrollable list of voices - click a row to select + preview instantly
        self.voice_list_frame = ctk.CTkScrollableFrame(panel, corner_radius=10)
        self.voice_list_frame.grid(row=4, column=0, padx=16, pady=(0, 12), sticky="nsew")
        self.voice_list_frame.grid_columnconfigure(0, weight=1)
        self.voice_buttons = {}  # voice_id -> CTkButton

        # Status / info box for the selected character - must exist before
        # _populate_voice_list() runs, since it updates this label.
        info_frame = ctk.CTkFrame(panel, corner_radius=10)
        info_frame.grid(row=5, column=0, padx=16, pady=(0, 12), sticky="ew")
        info_frame.grid_columnconfigure(0, weight=1)

        self.character_status_label = ctk.CTkLabel(
            info_frame,
            text="Select a character to preview.",
            font=ctk.CTkFont(size=self._f(13)),
            wraplength=self._f(260),
            justify="left",
        )
        self.character_status_label.grid(row=0, column=0, padx=12, pady=12, sticky="nw")

        self._populate_voice_list(self.current_lang_code)

        self.preview_progress = ctk.CTkProgressBar(panel, mode="indeterminate")
        self.preview_progress.grid(row=6, column=0, padx=16, pady=(0, 8), sticky="ew")
        self.preview_progress.set(0)

        create_voice_button = ctk.CTkButton(
            panel,
            text="+ Create New Voice",
            command=self._open_voice_lab,
            height=self._f(38),
            font=ctk.CTkFont(size=self._f(13), weight="bold"),
            fg_color="gray30",
            hover_color="gray20",
        )
        create_voice_button.grid(row=7, column=0, padx=16, pady=(0, 16), sticky="ew")

    def _open_voice_lab(self):
        VoiceLabWindow(
            self,
            engine=self.engine,
            chatterbox_bridge=self.chatterbox_bridge,
            registry=self.voice_registry,
            voices=VOICES,
            preview_texts=PREVIEW_TEXTS,
            sample_rate=SAMPLE_RATE,
            on_voice_created=self._on_custom_voice_created,
            scale=self.scale,
        )

    def _on_custom_voice_created(self):
        if self.current_lang_code == "custom":
            self.after(0, lambda: self._populate_voice_list("custom"))

    def _populate_voice_list(self, lang_code: str):
        """Clear and rebuild the scrollable voice list for a given language
        (or the custom "My Voices" list when lang_code == "custom")."""
        for widget in self.voice_list_frame.winfo_children():
            widget.destroy()
        self.voice_buttons = {}

        if lang_code == "custom":
            matching_voices = [
                (meta["display_name"], f"custom::{key}", meta.get("lang_code", "a"))
                for key, meta in self.voice_registry.all_voices().items()
            ]
            if not matching_voices:
                empty_label = ctk.CTkLabel(
                    self.voice_list_frame,
                    text="No custom voices yet.\nUse '+ Create New Voice' below to make one.",
                    font=ctk.CTkFont(size=self._f(12)),
                    text_color="gray60",
                    justify="left",
                )
                empty_label.grid(row=0, column=0, padx=8, pady=8, sticky="w")
                self.selected_voice_id = None
                self.selected_lang_code = "a"
                self.character_status_label.configure(text="No custom voices yet.")
                return
        else:
            matching_voices = [v for v in VOICES if v[2] == lang_code]

        for i, (display_name, voice_id, vlang_code) in enumerate(matching_voices):
            is_custom_voice = voice_id.startswith("custom::")
            if is_custom_voice:
                key = voice_id.split("custom::", 1)[1]
                has_preview = os.path.exists(os.path.join(CUSTOM_PREVIEW_DIR, f"{key}.wav"))
            else:
                has_preview = os.path.exists(
                    os.path.join(CHARACTER_SPEECH_DIR, f"{voice_id}.wav")
                )
            label_text = f"🔊 {display_name}" if has_preview else f"○ {display_name}"

            # Wrap the select button (and, for custom voices, a delete
            # button) in a per-row frame so the trash icon can sit alongside
            # it without disturbing the select button's own click handler.
            row = ctk.CTkFrame(self.voice_list_frame, fg_color="transparent")
            row.grid(row=i, column=0, padx=4, pady=3, sticky="ew")
            row.grid_columnconfigure(0, weight=1)

            btn = ctk.CTkButton(
                row,
                text=label_text,
                anchor="w",
                height=self._f(38),
                font=ctk.CTkFont(size=self._f(14)),
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray80", "gray30"),
                command=lambda vid=voice_id, lc=vlang_code, name=display_name: self._on_voice_row_clicked(
                    vid, lc, name
                ),
            )
            btn.grid(row=0, column=0, sticky="ew")
            self.voice_buttons[voice_id] = btn

            if is_custom_voice:
                del_btn = ctk.CTkButton(
                    row,
                    text="🗑",
                    width=self._f(32),
                    height=self._f(38),
                    font=ctk.CTkFont(size=self._f(14)),
                    fg_color="gray30",
                    hover_color="#8B2C2C",
                    command=lambda vid=voice_id, name=display_name: self._on_delete_custom_voice_clicked(
                        vid, name
                    ),
                )
                del_btn.grid(row=0, column=1, padx=(4, 0))

        # Auto-select the first voice in the newly shown language
        if matching_voices:
            first_name, first_id, first_lang = matching_voices[0]
            self.selected_voice_id = first_id
            self.selected_lang_code = first_lang
            self._highlight_selected_voice(first_id)
            self.character_status_label.configure(
                text=f"'{first_name}' selected.\nClick a name to preview."
            )

    def _on_language_selected(self, label: str):
        if label == "★ My Voices":
            self.current_lang_code = "custom"
            self.language_note_label.configure(text="")
            self._populate_voice_list("custom")
            return
        for lbl, code, note in LANGUAGES:
            if lbl == label:
                self.current_lang_code = code
                self.language_note_label.configure(text=note or "")
                self._populate_voice_list(code)
                return

    # -- Right panel: text input + generate -------------------------------
    def _build_text_panel(self):
        panel = ctk.CTkFrame(self, corner_radius=12)
        panel.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)
        panel.grid_rowconfigure(2, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        # -- Title row: title label + Help button side by side ------------
        title_row = ctk.CTkFrame(panel, fg_color="transparent")
        title_row.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        title_row.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            title_row, text="Text to Generate", font=ctk.CTkFont(size=self._f(20), weight="bold")
        )
        title.grid(row=0, column=0, sticky="w")

        help_button = ctk.CTkButton(
            title_row,
            text="? Help",
            width=self._f(70),
            height=self._f(30),
            font=ctk.CTkFont(size=self._f(12)),
            fg_color="gray30",
            hover_color="gray20",
            command=self._show_help_dialog,
        )
        help_button.grid(row=0, column=1, sticky="e")

        # -- Pause quick-insert button ----------------------------------
        filler_row = ctk.CTkFrame(panel, fg_color="transparent")
        filler_row.grid(row=1, column=0, padx=16, pady=(0, 4), sticky="w")

        filler_label = ctk.CTkLabel(
            filler_row, text="Insert:", font=ctk.CTkFont(size=self._f(12)), text_color="gray60"
        )
        filler_label.pack(side="left", padx=(0, 6))

        pause_btn = ctk.CTkButton(
            filler_row,
            text="pause",
            width=self._f(56),
            height=self._f(24),
            font=ctk.CTkFont(size=self._f(11)),
            fg_color="gray30",
            hover_color="gray20",
            command=self._insert_pause_marker,
        )
        pause_btn.pack(side="left", padx=(0, 4))

        self.text_box = ctk.CTkTextbox(
            panel,
            font=ctk.CTkFont(size=self._f(15)),
            wrap="word",
        )
        self.text_box.grid(row=2, column=0, padx=16, pady=(0, 12), sticky="nsew")
        self.text_box.insert("1.0", self._load_last_text())

        # Auto-save the text box contents shortly after typing stops, and
        # whenever the window loses focus, so nothing is lost.
        self.text_box.bind("<KeyRelease>", self._on_text_changed)
        self.text_box.bind("<FocusOut>", lambda e: self._save_last_text())

        filename_row = ctk.CTkFrame(panel, fg_color="transparent")
        filename_row.grid(row=3, column=0, padx=16, pady=(0, 8), sticky="ew")
        filename_row.grid_columnconfigure(0, weight=1)

        filename_label = ctk.CTkLabel(
            filename_row, text="Output file name:", font=ctk.CTkFont(size=self._f(13))
        )
        filename_label.grid(row=0, column=0, sticky="w")

        self.filename_entry = ctk.CTkEntry(
            filename_row,
            placeholder_text="e.g. intro_narration",
            font=ctk.CTkFont(size=self._f(14)),
            height=self._f(34),
        )
        self.filename_entry.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        controls = ctk.CTkFrame(panel, fg_color="transparent")
        controls.grid(row=4, column=0, padx=16, pady=(0, 8), sticky="ew")
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=0)

        speed_label = ctk.CTkLabel(controls, text="Speed:", font=ctk.CTkFont(size=self._f(13)))
        speed_label.grid(row=0, column=0, sticky="w")

        self.speed_slider = ctk.CTkSlider(controls, from_=0.5, to=1.8, number_of_steps=13)
        self.speed_slider.set(1.0)
        self.speed_slider.grid(row=1, column=0, sticky="ew", padx=(0, 12))

        self.speed_value_label = ctk.CTkLabel(
            controls, text="1.0x", font=ctk.CTkFont(size=self._f(13))
        )
        self.speed_value_label.grid(row=1, column=1, sticky="e")
        self.speed_slider.configure(
            command=lambda v: self.speed_value_label.configure(text=f"{v:.1f}x")
        )

        # -- Emotion intensity slider (Chatterbox only) ----------------------
        intensity_label = ctk.CTkLabel(
            controls, text="Emotion intensity (Chatterbox only):",
            font=ctk.CTkFont(size=self._f(13)),
        )
        intensity_label.grid(row=2, column=0, sticky="w", pady=(10, 0))

        self.intensity_slider = ctk.CTkSlider(controls, from_=0.1, to=1.5, number_of_steps=14)
        self.intensity_slider.set(0.5)
        self.intensity_slider.grid(row=3, column=0, sticky="ew", padx=(0, 12))

        self.intensity_value_label = ctk.CTkLabel(
            controls, text="0.50", font=ctk.CTkFont(size=self._f(13))
        )
        self.intensity_value_label.grid(row=3, column=1, sticky="e")
        self.intensity_slider.configure(
            command=lambda v: self.intensity_value_label.configure(text=f"{v:.2f}")
        )

        # -- Emotion tag toggles --------------------------------------------
        emotions_frame = ctk.CTkFrame(panel, corner_radius=10)
        emotions_frame.grid(row=5, column=0, padx=16, pady=(0, 8), sticky="ew")
        for col in range(3):
            emotions_frame.grid_columnconfigure(col, weight=1)

        emotions_title = ctk.CTkLabel(
            emotions_frame,
            text='Emotions - type "[tag]" in your text, e.g. "Ha! [laugh]" (click Help for details)',
            font=ctk.CTkFont(size=self._f(12), weight="bold"),
            wraplength=self._f(680),
            justify="left",
        )
        emotions_title.grid(row=0, column=0, columnspan=3, padx=12, pady=(10, 4), sticky="w")

        self.emotion_switches = {}
        for idx, tag in enumerate(EMOTION_TAGS):
            row = 1 + idx // 3
            col = idx % 3
            switch = ctk.CTkSwitch(
                emotions_frame,
                text=tag,
                font=ctk.CTkFont(size=self._f(12)),
            )
            switch.select()  # enabled by default
            switch.grid(row=row, column=col, padx=12, pady=4, sticky="w")
            self.emotion_switches[tag] = switch

        action_row = ctk.CTkFrame(panel, fg_color="transparent")
        action_row.grid(row=6, column=0, padx=16, pady=(4, 8), sticky="ew")
        action_row.grid_columnconfigure(0, weight=3)
        action_row.grid_columnconfigure(1, weight=1)

        self.generate_button = ctk.CTkButton(
            action_row,
            text="🎙 Generate Sound",
            command=self._on_generate_clicked,
            height=self._f(48),
            font=ctk.CTkFont(size=self._f(17), weight="bold"),
        )
        self.generate_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.stop_button = ctk.CTkButton(
            action_row,
            text="🔇 Stop",
            command=self._on_stop_clicked,
            height=self._f(48),
            font=ctk.CTkFont(size=self._f(17), weight="bold"),
            fg_color="#8B2C2C",
            hover_color="#6E2222",
        )
        self.stop_button.grid(row=0, column=1, sticky="ew")

        self.generate_progress = ctk.CTkProgressBar(panel, mode="indeterminate")
        self.generate_progress.grid(row=7, column=0, padx=16, pady=(0, 8), sticky="ew")
        self.generate_progress.set(0)

        self.output_status_label = ctk.CTkLabel(
            panel,
            text="Generated audio will appear here.",
            font=ctk.CTkFont(size=self._f(13)),
            wraplength=self._f(700),
            justify="left",
            anchor="w",
        )
        self.output_status_label.grid(row=8, column=0, padx=16, pady=(0, 16), sticky="ew")

    def _insert_pause_marker(self):
        # Inserts a default 0.5s pause - edit the number directly in the
        # text box to change the duration, e.g. change to {pause:1.2}.
        self.text_box.insert("insert", " {pause:0.5} ")
        self.text_box.focus_set()

    # ------------------------------------------------------------------
    # Help dialog
    # ------------------------------------------------------------------
    def _show_help_dialog(self):
        win = ctk.CTkToplevel(self)
        win.title("Help & Emotion Guide")
        win.geometry(f"{self._f(560)}x{self._f(640)}")
        win.grab_set()

        scroll = ctk.CTkScrollableFrame(win, corner_radius=0)
        scroll.pack(fill="both", expand=True, padx=16, pady=16)

        def add_section(title_text, body_text):
            t = ctk.CTkLabel(
                scroll, text=title_text,
                font=ctk.CTkFont(size=self._f(15), weight="bold"),
                anchor="w", justify="left",
            )
            t.pack(fill="x", pady=(10, 2))
            b = ctk.CTkLabel(
                scroll, text=body_text,
                font=ctk.CTkFont(size=self._f(12)),
                anchor="w", justify="left", wraplength=self._f(480),
            )
            b.pack(fill="x", pady=(0, 4))

        add_section(
            "How this works",
            "Kokoro handles normal narration - fast, and used by default. When "
            "your text contains an enabled emotion tag like [laugh], that "
            "generation is routed to Chatterbox instead, which clones the "
            "current voice and performs the reaction for real, rather than "
            "just reading the word."
        )
        add_section(
            "Available emotion tags",
            "[laugh]  [chuckle]  [cough]  [sigh]  [clear throat]  [shush]  "
            "[groan]  [sniff]  [gasp]\n\n"
            "Type them directly into your script, for example:\n"
            "\"That's hilarious! [laugh] I can't believe it.\"\n"
            "\"Well, [chuckle] I suppose that's one way to do it.\""
        )
        add_section(
            "Emotion intensity",
            "The Emotion Intensity slider controls how strongly Chatterbox "
            "leans into the emotional/expressive character of a tagged line. "
            "Lower values stay closer to your reference voice; higher values "
            "produce a bigger, more exaggerated reaction. Only affects lines "
            "routed to Chatterbox - plain Kokoro narration is unaffected."
        )
        add_section(
            "Pauses ({pause:N})",
            "Type or insert \"{pause:N}\" to get an exact, controllable gap "
            "of N seconds of real silence at that point - e.g. \"{pause:0.6}\" "
            "for 600ms. This is genuine dead air, not something the TTS "
            "engine has to guess from a comma, so it's the most reliable way "
            "to control pacing. The Insert 'pause' button drops in "
            "\"{pause:0.5}\" - just edit the number directly in the text box "
            "to change the length. Pauses are capped at 10 seconds each so a "
            "typo can't stall generation."
        )
        add_section(
            "Enabling / disabling tags",
            "Each tag has its own switch above the Generate button, ON by "
            "default. Turning a tag OFF strips that specific tag out of your "
            "text before generating - so a disabled [cough] is silently "
            "removed and won't trigger Chatterbox on its own. If every tag in "
            "your text ends up disabled, generation falls back to fast, "
            "Kokoro-only narration automatically."
        )
        add_section(
            "Known limitations",
            "Emotion tags are a genuine trained feature, but Chatterbox's own "
            "team has described them as hit-or-miss - not every tag will land "
            "every time. Chatterbox generation is also noticeably slower than "
            "Kokoro, and requires chatterbox-env to be installed alongside "
            "this app (see install_chatterbox.sh)."
        )
        add_section(
            "Example prompts to try",
            "\"Oh man, [laugh] that's such a good story.\"\n"
            "\"Sorry, excuse me, [cough] where was I?\"\n"
            "\"[sigh] Fine, I'll take care of it myself.\"\n"
            "\"Okay, {pause:0.4} let me think about this. {pause:0.6} Alright.\"\n"
            "\"Wait, [shush] do you hear that? [gasp] Never mind, it's just "
            "the mailman.\""
        )
        add_section(
            "Managing custom voices",
            "Custom voices you create in '+ Create New Voice' show up under "
            "the '★ My Voices' language entry. Each row there has a trash "
            "icon next to it - click it and confirm to permanently delete "
            "that voice, its saved preview clip, and its cached reference "
            "audio."
        )

        close_btn = ctk.CTkButton(win, text="Close", command=win.destroy)
        close_btn.pack(pady=(0, 16))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _voice_lookup(self, display_name: str):
        for name, voice_id, lang_code in VOICES:
            if name == display_name:
                return voice_id, lang_code
        return VOICES[0][1], VOICES[0][2]

    def _highlight_selected_voice(self, voice_id: str):
        """Visually mark the selected row and reset the others."""
        for vid, btn in self.voice_buttons.items():
            if vid == voice_id:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color="transparent")

    def _on_voice_row_clicked(self, voice_id: str, lang_code: str, display_name: str):
        self.selected_voice_id = voice_id
        self.selected_lang_code = lang_code
        self._highlight_selected_voice(voice_id)

        is_custom = voice_id.startswith("custom::")
        custom_meta = None
        if is_custom:
            custom_key = voice_id.split("custom::", 1)[1]
            custom_meta = self.voice_registry.get(custom_key)
            os.makedirs(CUSTOM_PREVIEW_DIR, exist_ok=True)
            sample_path = os.path.join(CUSTOM_PREVIEW_DIR, f"{custom_key}.wav")
        else:
            sample_path = os.path.join(CHARACTER_SPEECH_DIR, f"{voice_id}.wav")

        for btn in self.voice_buttons.values():
            btn.configure(state="disabled")
        self.preview_progress.start()

        def worker():
            try:
                if os.path.exists(sample_path):
                    self._set_label(
                        self.character_status_label,
                        f"'{display_name}' selected.\nPlaying saved preview...",
                    )
                else:
                    self._set_label(
                        self.character_status_label,
                        f"'{display_name}' selected.\nGenerating preview...",
                    )
                    if is_custom and custom_meta and custom_meta["type"] == "clone":
                        audio = self.chatterbox_bridge.synthesize_chunk_with_reference(
                            PREVIEW_TEXTS["a"],
                            custom_meta["reference_path"],
                            exaggeration=custom_meta.get("exaggeration", 0.5),
                            seed=custom_meta.get("seed"),
                            cfg_weight=custom_meta.get("cfg_weight", 0.5),
                            temperature=custom_meta.get("temperature", 0.8),
                        )
                    elif is_custom and custom_meta and custom_meta["type"] == "mix":
                        mix_lang = custom_meta.get("lang_code", "a")
                        audio = self.engine.synthesize(
                            PREVIEW_TEXTS.get(mix_lang, PREVIEW_TEXTS["a"]),
                            custom_meta["voice_id"],
                            mix_lang,
                        )
                    else:
                        audio = self.engine.synthesize(
                            PREVIEW_TEXTS.get(lang_code, "Hi there! This is a quick preview."),
                            voice_id,
                            lang_code,
                        )
                    sf.write(sample_path, audio, SAMPLE_RATE)
                    if not is_custom:
                        self.after(
                            0,
                            lambda: self.voice_buttons[voice_id].configure(
                                text=f"🔊 {display_name}"
                            ),
                        )
                    self._set_label(
                        self.character_status_label,
                        f"'{display_name}' selected.\nPreview saved and ready.",
                    )

                self._play_audio(sample_path)
            except Exception as e:
                self._set_label(self.character_status_label, f"Error: {e}")
            finally:
                self.after(0, self.preview_progress.stop)
                self.after(0, lambda: [b.configure(state="normal") for b in self.voice_buttons.values()])

        threading.Thread(target=worker, daemon=True).start()

    def _on_delete_custom_voice_clicked(self, voice_id: str, display_name: str):
        """Shows a confirm dialog, then deletes the custom voice if the user
        confirms. voice_id is the "custom::<key>" id used in self.voice_buttons."""
        key = voice_id.split("custom::", 1)[1]

        dialog = ctk.CTkToplevel(self)
        dialog.title("Delete Voice")
        dialog.geometry(f"{self._f(360)}x{self._f(170)}")
        dialog.grab_set()
        dialog.resizable(False, False)

        msg = ctk.CTkLabel(
            dialog,
            text=f"Delete '{display_name}'?\nThis removes the voice and its saved "
                 f"audio, and can't be undone.",
            font=ctk.CTkFont(size=self._f(13)),
            wraplength=self._f(320),
            justify="left",
        )
        msg.pack(padx=16, pady=(16, 12))

        btn_row = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_row.pack(padx=16, pady=(0, 16), fill="x")
        btn_row.grid_columnconfigure((0, 1), weight=1)

        cancel_btn = ctk.CTkButton(
            btn_row, text="Cancel", fg_color="gray30", hover_color="gray20",
            command=dialog.destroy,
        )
        cancel_btn.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        def do_delete():
            dialog.destroy()
            self._delete_custom_voice(key, display_name)

        confirm_btn = ctk.CTkButton(
            btn_row, text="Delete", fg_color="#8B2C2C", hover_color="#6E2222",
            command=do_delete,
        )
        confirm_btn.grid(row=0, column=1, padx=(6, 0), sticky="ew")

    def _delete_custom_voice(self, key: str, display_name: str):
        """Removes a custom voice from the registry and cleans up any
        cached audio (preview clip + Chatterbox reference clip) tied to it."""
        try:
            self.voice_registry.delete(key)
        except AttributeError:
            self._set_label(
                self.character_status_label,
                "Couldn't delete: VoiceRegistry has no delete() method yet.",
            )
            return
        except Exception as e:
            self._set_label(self.character_status_label, f"Error deleting voice: {e}")
            return

        preview_path = os.path.join(CUSTOM_PREVIEW_DIR, f"{key}.wav")
        if os.path.exists(preview_path):
            try:
                os.remove(preview_path)
            except Exception as e:
                print(f"Could not remove preview file: {e}")

        ref_path = self.chatterbox_bridge._reference_clip_path(key)
        if os.path.exists(ref_path):
            try:
                os.remove(ref_path)
            except Exception as e:
                print(f"Could not remove reference clip: {e}")

        # If the voice we just deleted was the selected one, clear that state
        # so generation doesn't try to use a voice that no longer exists.
        deleted_voice_id = f"custom::{key}"
        if getattr(self, "selected_voice_id", None) == deleted_voice_id:
            self.selected_voice_id = None

        if self.current_lang_code == "custom":
            self._populate_voice_list("custom")

        self.character_status_label.configure(text=f"'{display_name}' deleted.")

    def _on_generate_clicked(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            self._set_label(self.output_status_label, "Please paste or type some text first.")
            return

        voice_id = self.selected_voice_id
        if not voice_id:
            self._set_label(self.output_status_label, "Select or create a voice first.")
            return

        self._save_last_text()

        lang_code = self.selected_lang_code
        speed = float(self.speed_slider.get())
        exaggeration = float(self.intensity_slider.get())

        # Custom voices (from the "Create New Voice" panel) are prefixed
        # "custom::<key>" - resolve what kind it is up front so the loop
        # below knows how to route each group.
        is_custom = voice_id.startswith("custom::")
        custom_key = voice_id.split("custom::", 1)[1] if is_custom else None
        custom_meta = self.voice_registry.get(custom_key) if is_custom else None
        is_clone = bool(custom_meta and custom_meta.get("type") == "clone")
        is_mix = bool(custom_meta and custom_meta.get("type") == "mix")
        if is_mix:
            lang_code = custom_meta.get("lang_code", lang_code)

        # A cloned voice pins the seed you picked in the lab so its accent
        # stays consistent between generations, plus its delivery knobs.
        clone_seed = custom_meta.get("seed") if is_clone else None
        clone_cfg = custom_meta.get("cfg_weight", 0.5) if is_clone else 0.5
        clone_temp = custom_meta.get("temperature", 0.8) if is_clone else 0.8

        enabled_tags = {
            tag for tag, switch in self.emotion_switches.items() if switch.get() == 1
        }

        self.generate_button.configure(state="disabled")
        self.generate_progress.start()

        def worker():
            try:
                items = split_into_chunks(text)
                groups = group_chunks_by_engine(items, enabled_tags)
                if not groups:
                    raise RuntimeError("Nothing to generate.")

                audio_segments = []
                total = len(groups)

                for i, group in enumerate(groups, start=1):
                    engine = group["engine"]

                    if engine == "pause":
                        seconds = group["seconds"]
                        self._set_label(
                            self.output_status_label,
                            f"Group {i}/{total}: inserting a {seconds:.2f}s pause...",
                        )
                        audio_segments.append(silence(seconds))
                        continue

                    if is_clone:
                        # A cloned voice has no Kokoro identity to fall back
                        # on - every line (tagged or not) goes through
                        # Chatterbox using the imported reference recording.
                        self._set_label(
                            self.output_status_label,
                            f"Group {i}/{total}: generating with your cloned voice...",
                        )
                        seg = self.chatterbox_bridge.synthesize_chunk_with_reference(
                            group["text"], custom_meta["reference_path"],
                            exaggeration=exaggeration, seed=clone_seed,
                            cfg_weight=clone_cfg, temperature=clone_temp,
                        )
                    elif engine == "chatterbox":
                        self._set_label(
                            self.output_status_label,
                            f"Group {i}/{total}: generating with emotion (Chatterbox)...",
                        )
                        synth_voice = custom_meta["voice_id"] if is_mix else voice_id
                        seg = self.chatterbox_bridge.synthesize_chunk(
                            group["text"], synth_voice, lang_code,
                            exaggeration=exaggeration,
                            reference_key=custom_key if is_mix else None,
                        )
                    else:
                        self._set_label(
                            self.output_status_label,
                            f"Group {i}/{total}: generating (Kokoro)...",
                        )
                        synth_voice = custom_meta["voice_id"] if is_mix else voice_id
                        seg = self.engine.synthesize(
                            group["text"], synth_voice, lang_code, speed=speed
                        )

                    audio_segments.append(seg)
                    audio_segments.append(silence(0.12))  # shorter pause between groups

                final_audio = np.concatenate(audio_segments)
                out_path_label = custom_key if is_custom else voice_id
                out_path = self._build_output_path(out_path_label)
                sf.write(out_path, final_audio, SAMPLE_RATE)
                duration = len(final_audio) / SAMPLE_RATE

                self._set_label(
                    self.output_status_label,
                    f"Done! Saved to {out_path} ({duration:.1f}s, {total} groups merged). Playing now...",
                )
                self._play_audio(out_path)
            except Exception as e:
                self._set_label(self.output_status_label, f"Error: {e}")
            finally:
                self.after(0, self.generate_progress.stop)
                self.after(0, lambda: self.generate_button.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _build_output_path(self, voice_id: str) -> str:
        """Build the save path, using the user's filename if given (sanitized,
        with auto-numbering to avoid overwriting), else a timestamp default."""
        raw_name = self.filename_entry.get().strip()

        if raw_name:
            # Strip a trailing .wav the user may have typed, then sanitize.
            if raw_name.lower().endswith(".wav"):
                raw_name = raw_name[:-4]
            safe_name = "".join(
                c if (c.isalnum() or c in ("_", "-", " ")) else "_" for c in raw_name
            ).strip().replace(" ", "_")
            if not safe_name:
                safe_name = None
        else:
            safe_name = None

        if safe_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{timestamp}_{voice_id}"
        else:
            base_name = safe_name

        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.wav")

        # Avoid silently overwriting an existing file with the same name.
        counter = 2
        while os.path.exists(out_path):
            out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{counter}.wav")
            counter += 1

        return out_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _on_stop_clicked(self):
        try:
            pygame.mixer.music.stop()
            self._set_label(self.output_status_label, "Playback stopped.")
        except Exception as e:
            print(f"Stop error: {e}")

    def _set_label(self, label: ctk.CTkLabel, text: str):
        self.after(0, lambda: label.configure(text=text))

    def _play_audio(self, path: str):
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Playback error: {e}")

    # ------------------------------------------------------------------
    # Text persistence - remembers the last pasted/typed text across runs
    # ------------------------------------------------------------------
    def _load_last_text(self) -> str:
        if os.path.exists(LAST_TEXT_FILE):
            try:
                with open(LAST_TEXT_FILE, "r", encoding="utf-8") as f:
                    saved = f.read()
                if saved.strip():
                    return saved
            except Exception as e:
                print(f"Could not load last text: {e}")
        return "Paste or type the text you want turned into speech here."

    def _save_last_text(self):
        try:
            text = self.text_box.get("1.0", "end-1c")
            with open(LAST_TEXT_FILE, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"Could not save last text: {e}")

    def _on_text_changed(self, event=None):
        # Debounce: cancel any pending save and schedule a new one, so we
        # only write to disk ~1s after the user stops typing.
        if self._save_after_id is not None:
            self.after_cancel(self._save_after_id)
        self._save_after_id = self.after(1000, self._save_last_text)

    def _on_close(self):
        self._save_last_text()
        self.destroy()


def _split_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_into_chunks(text: str):
    """Tokenizes text into an ordered list of items:
      {"type": "sentence", "text": "..."}     - normal narration
      {"type": "pause", "seconds": 0.6}       - a {pause:N} marker

    Markers are pulled out first so they survive as their own ordered token
    instead of getting mangled by sentence splitting; group_chunks_by_engine
    then decides what engine/audio each item maps to.
    """
    items = []
    for para in text.split("\n"):
        if not para.strip():
            continue

        pos = 0
        for m in MARKER_RE.finditer(para):
            before = para[pos:m.start()]
            for s in _split_sentences(before):
                items.append({"type": "sentence", "text": s})

            seconds = max(0.0, min(float(m.group(1)), MAX_PAUSE_SECONDS))
            items.append({"type": "pause", "seconds": seconds})

            pos = m.end()

        remainder = para[pos:]
        for s in _split_sentences(remainder):
            items.append({"type": "sentence", "text": s})

    return items


def group_chunks_by_engine(items, enabled_tags, max_chars=280):
    """Groups consecutive same-engine sentence items into single generation
    calls (so prosody flows naturally within each group). Still splits on
    engine changes, on a {pause:N} marker, or when a group gets too long
    (Chatterbox degrades on very long single passes)."""
    groups = []
    current_text = []
    current_engine = None

    def flush():
        if current_text:
            groups.append({"engine": current_engine, "text": " ".join(current_text)})

    for item in items:
        if item["type"] == "pause":
            flush()
            current_text = []
            current_engine = None
            groups.append({"engine": "pause", "seconds": item["seconds"]})
            continue

        filtered = strip_disabled_tags(item["text"], enabled_tags)
        if not filtered.strip():
            continue

        engine = "chatterbox" if text_has_any_tag(filtered) else "kokoro"
        joined_len = sum(len(s) for s in current_text) + len(filtered)

        if engine != current_engine or joined_len > max_chars:
            flush()
            current_text = [filtered]
            current_engine = engine
        else:
            current_text.append(filtered)

    flush()
    return groups


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def silence(duration_sec: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(duration_sec * sr), dtype=np.float32)


if __name__ == "__main__":
    app = CharacterVoiceStudio()
    app.mainloop()