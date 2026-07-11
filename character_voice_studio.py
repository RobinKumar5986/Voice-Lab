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
5. Help button explaining all of the above with example prompts.

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

    def synthesize(self, text: str, voice_id: str, lang_code: str, speed: float = 1.0):
        """Returns concatenated float32 numpy audio at SAMPLE_RATE."""
        import numpy as np

        pipeline = self.get_pipeline(lang_code)
        chunks = []
        generator = pipeline(text, voice=voice_id, speed=speed, split_pattern=r"\n+")
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

    def _reference_clip_path(self, voice_id: str) -> str:
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        return os.path.join(REFERENCE_DIR, f"{voice_id}_reference.wav")

    def ensure_reference_clip(self, voice_id: str, lang_code: str) -> str:
        ref_path = self._reference_clip_path(voice_id)
        if os.path.exists(ref_path):
            return ref_path
        audio = self.kokoro_engine.synthesize(REFERENCE_TEXT, voice_id, lang_code, speed=1.0)
        sf.write(ref_path, audio, SAMPLE_RATE)
        return ref_path

    def synthesize_chunk(self, text: str, voice_id: str, lang_code: str) -> np.ndarray:
        """Generates ONE short chunk via Chatterbox and returns audio as a
        numpy array resampled to SAMPLE_RATE, ready to concatenate with
        Kokoro chunks."""
        if not os.path.exists(CHATTERBOX_ENV_PYTHON):
            raise RuntimeError(
                "Chatterbox isn't installed (chatterbox-env not found next to this "
                "script). Disable emotion tags to use fast Kokoro-only generation, "
                "or run install_chatterbox.sh first."
            )

        reference_clip = self.ensure_reference_clip(voice_id, lang_code)
        tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        try:
            result = subprocess.run(
                [
                    CHATTERBOX_ENV_PYTHON,
                    CHATTERBOX_WORKER_SCRIPT,
                    "--text", text,
                    "--reference", reference_clip,
                    "--output", tmp_path,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Chatterbox generation failed:\n{result.stderr.strip()}")

            audio, sr = sf.read(tmp_path, dtype="float32")
            print(f"[DEBUG] Chatterbox output sr={sr}, len={len(audio)}")
            return resample_audio(audio, sr, SAMPLE_RATE)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                

class CharacterVoiceStudio(ctk.CTk):
    def __init__(self):
        super().__init__()

        os.makedirs(CHARACTER_SPEECH_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.engine = VoiceEngine()
        self.chatterbox_bridge = ChatterboxBridge(self.engine)
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

        # Language selector - filters which voices show up below
        language_labels = [lbl for lbl, _, _ in LANGUAGES]
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
        self.preview_progress.grid(row=6, column=0, padx=16, pady=(0, 16), sticky="ew")
        self.preview_progress.set(0)

    def _populate_voice_list(self, lang_code: str):
        """Clear and rebuild the scrollable voice list for a given language."""
        for widget in self.voice_list_frame.winfo_children():
            widget.destroy()
        self.voice_buttons = {}

        matching_voices = [v for v in VOICES if v[2] == lang_code]
        for i, (display_name, voice_id, vlang_code) in enumerate(matching_voices):
            has_preview = os.path.exists(
                os.path.join(CHARACTER_SPEECH_DIR, f"{voice_id}.wav")
            )
            label_text = f"🔊 {display_name}" if has_preview else f"○ {display_name}"
            btn = ctk.CTkButton(
                self.voice_list_frame,
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
            btn.grid(row=i, column=0, padx=4, pady=3, sticky="ew")
            self.voice_buttons[voice_id] = btn

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
        panel.grid_rowconfigure(1, weight=1)
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

        self.text_box = ctk.CTkTextbox(
            panel,
            font=ctk.CTkFont(size=self._f(15)),
            wrap="word",
        )
        self.text_box.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="nsew")
        self.text_box.insert("1.0", self._load_last_text())

        # Auto-save the text box contents shortly after typing stops, and
        # whenever the window loses focus, so nothing is lost.
        self.text_box.bind("<KeyRelease>", self._on_text_changed)
        self.text_box.bind("<FocusOut>", lambda e: self._save_last_text())

        filename_row = ctk.CTkFrame(panel, fg_color="transparent")
        filename_row.grid(row=2, column=0, padx=16, pady=(0, 8), sticky="ew")
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
        controls.grid(row=3, column=0, padx=16, pady=(0, 8), sticky="ew")
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

        # -- Emotion tag toggles --------------------------------------------
        emotions_frame = ctk.CTkFrame(panel, corner_radius=10)
        emotions_frame.grid(row=4, column=0, padx=16, pady=(0, 8), sticky="ew")
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
        action_row.grid(row=5, column=0, padx=16, pady=(4, 8), sticky="ew")
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
        self.generate_progress.grid(row=6, column=0, padx=16, pady=(0, 8), sticky="ew")
        self.generate_progress.set(0)

        self.output_status_label = ctk.CTkLabel(
            panel,
            text="Generated audio will appear here.",
            font=ctk.CTkFont(size=self._f(13)),
            wraplength=self._f(700),
            justify="left",
            anchor="w",
        )
        self.output_status_label.grid(row=7, column=0, padx=16, pady=(0, 16), sticky="ew")

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
            "\"Wait, [shush] do you hear that? [gasp] Never mind, it's just "
            "the mailman.\""
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
                    audio = self.engine.synthesize(
                        PREVIEW_TEXTS.get(lang_code, "Hi there! This is a quick preview."),
                        voice_id,
                        lang_code,
                    )
                    sf.write(sample_path, audio, SAMPLE_RATE)
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

    
    def _on_generate_clicked(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            self._set_label(self.output_status_label, "Please paste or type some text first.")
            return

        self._save_last_text()

        voice_id = self.selected_voice_id
        lang_code = self.selected_lang_code
        speed = float(self.speed_slider.get())

        enabled_tags = {
            tag for tag, switch in self.emotion_switches.items() if switch.get() == 1
        }

        self.generate_button.configure(state="disabled")
        self.generate_progress.start()

        def worker():
            try:
                sentences = split_into_chunks(text)
                groups = group_chunks_by_engine(sentences, enabled_tags)
                if not groups:
                    raise RuntimeError("Nothing to generate.")

                audio_segments = []
                total = len(groups)

                for i, (engine, group_text) in enumerate(groups, start=1):
                    if engine == "chatterbox":
                        self._set_label(
                            self.output_status_label,
                            f"Group {i}/{total}: generating with emotion (Chatterbox)...",
                        )
                        seg = self.chatterbox_bridge.synthesize_chunk(
                            group_text, voice_id, lang_code
                        )
                    else:
                        self._set_label(
                            self.output_status_label,
                            f"Group {i}/{total}: generating (Kokoro)...",
                        )
                        seg = self.engine.synthesize(
                            group_text, voice_id, lang_code, speed=speed
                        )

                    audio_segments.append(seg)
                    audio_segments.append(silence(0.12))  # shorter pause between groups

                final_audio = np.concatenate(audio_segments)
                out_path = self._build_output_path(voice_id)
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

def split_into_chunks(text: str):
    """Splits text into sentence-level pieces. group_chunks_by_engine then
    merges consecutive same-engine sentences back together, so this is just
    the tokenizing step, not the final chunking."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for s in sentences:
            s = s.strip()
            if s:
                chunks.append(s)
    return chunks

def group_chunks_by_engine(sentences, enabled_tags, max_chars=280):
    """Groups consecutive sentences that use the same engine into single
    generation calls, so prosody flows naturally within each group instead
    of resetting every sentence. Still splits on engine changes or when a
    group gets too long (Chatterbox degrades on very long single passes)."""
    groups = []
    current_text = []
    current_engine = None

    def flush():
        if current_text:
            groups.append((current_engine, " ".join(current_text)))

    for sentence in sentences:
        filtered = strip_disabled_tags(sentence, enabled_tags)
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