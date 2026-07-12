"""
voice_lab.py - "Create New Voice" panel for Character Voice Studio.

Two ways to make a new, saveable custom voice:
  1. Mix - blend two bundled Kokoro voices together with a chosen balance.
     Kokoro is deterministic, so a given blend + text always sounds the same.
  2. Clone - import a real recording (yours or anyone with permission,
     5+ seconds, clean audio). The import is decoded and normalised to a
     clean mono WAV, then used as a Chatterbox reference clip.

Chatterbox generation is *stochastic*: with no fixed seed, every preview is a
different "take" (which is why the accent seems to drift between clicks). This
panel therefore exposes the controls that actually drive that variation:
  - Exaggeration (how strongly the expressive character comes through).
  - Seed (fix it for reproducible output, or randomise each preview).
Every preview is logged as a "Take" you can replay and pick from; the take you
select is pinned (its seed) onto the saved clone voice so it stays consistent
at generation time.

Custom voices are stored in characterSpeech/custom_voices/custom_voices.json
and picked up by the main app's "★ My Voices" list.
"""

import os
import json
import random
import shutil
import subprocess
import tempfile
import threading
import time
from tkinter import filedialog

import customtkinter as ctk
import soundfile as sf

CHARACTER_SPEECH_DIR = "characterSpeech"
CUSTOM_VOICES_DIR = os.path.join(CHARACTER_SPEECH_DIR, "custom_voices")
CUSTOM_VOICES_JSON = os.path.join(CUSTOM_VOICES_DIR, "custom_voices.json")
CUSTOM_REFERENCES_DIR = os.path.join(CUSTOM_VOICES_DIR, "references")
CUSTOM_PREVIEW_DIR = os.path.join(CUSTOM_VOICES_DIR, "previews")
LAST_IMPORT_DIR_FILE = os.path.join(CUSTOM_VOICES_DIR, "last_import_dir.txt")

MIN_CLONE_SECONDS = 5.0

# Normalised reference clips are written mono at this rate. Chatterbox
# resamples internally, so this just needs to be a clean, decodable wav.
NORMALIZED_SR = 24000

# How many recent preview takes to keep visible in the list.
MAX_TAKES = 10

# Seeds are drawn from this range (fits a 32-bit signed int for torch).
SEED_MAX = 2 ** 31 - 1

# Clone "style" presets. These set the Chatterbox delivery knobs only - they
# do NOT set the accent (accent always comes from the imported recording).
# The country-labelled ones simply bias toward faithful reproduction, which is
# what best preserves whatever accent is in the clip.
#   exaggeration - expressiveness   cfg_weight - guidance/pacing (low=looser)
#   temperature  - variety between takes
CLONE_PRESETS = {
    "Custom": None,  # leaves the sliders wherever the user put them
    "Faithful (1:1 clone)": {"exaggeration": 0.3, "cfg_weight": 0.7, "temperature": 0.6},
    "Soft clone":           {"exaggeration": 0.4, "cfg_weight": 0.5, "temperature": 0.7},
    "Standard clone":       {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
    "Expressive":           {"exaggeration": 0.8, "cfg_weight": 0.3, "temperature": 0.9},
    "US English (neutral)": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.7},
    "British English":      {"exaggeration": 0.45, "cfg_weight": 0.6, "temperature": 0.65},
    "Indian English":       {"exaggeration": 0.4, "cfg_weight": 0.65, "temperature": 0.6},
}

# Extra multiplier on top of the main window's scale factor, so this panel's
# text and controls read larger / easier to work with than the main window.
LAB_SCALE_BOOST = 1.25


# ---------------------------------------------------------------------------
# Registry - persists custom voices to disk between runs
# ---------------------------------------------------------------------------

class VoiceRegistry:
    def __init__(self):
        os.makedirs(CUSTOM_VOICES_DIR, exist_ok=True)
        os.makedirs(CUSTOM_REFERENCES_DIR, exist_ok=True)
        os.makedirs(CUSTOM_PREVIEW_DIR, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(CUSTOM_VOICES_JSON):
            try:
                with open(CUSTOM_VOICES_JSON, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Could not load custom voices: {e}")
        return {}

    def _save(self):
        try:
            with open(CUSTOM_VOICES_JSON, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"Could not save custom voices: {e}")

    def all_voices(self) -> dict:
        return dict(self._data)

    def get(self, key: str):
        return self._data.get(key)

    def _make_key(self, name: str) -> str:
        base = "".join(
            c if (c.isalnum() or c in ("_", "-")) else "_" for c in name.strip().lower()
        ).strip("_")
        if not base:
            base = "custom_voice"
        key = base
        counter = 2
        while key in self._data:
            key = f"{base}_{counter}"
            counter += 1
        return key

    def add_mix_voice(self, name: str, mix_voice_id: str, lang_code: str) -> str:
        """`mix_voice_id` is the blend recipe, e.g.
        'af_bella(0.70),af_sky(0.30)'. Resolved to a real voice tensor at
        generation time by VoiceEngine.synthesize."""
        key = self._make_key(name)
        self._data[key] = {
            "display_name": name,
            "type": "mix",
            "lang_code": lang_code,
            "voice_id": mix_voice_id,
        }
        self._save()
        return key

    def add_clone_voice(
        self,
        name: str,
        source_audio_path: str,
        seed=None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> str:
        """`source_audio_path` should already be a clean, decodable WAV (see
        VoiceLabWindow._normalize_to_wav) so Chatterbox can load it directly.

        `seed` (if given) is pinned onto the voice so every future generation
        reproduces the take you picked, instead of drifting each run.
        `exaggeration`/`cfg_weight`/`temperature` are stored as the voice's
        default delivery settings. (Accent itself comes from the recording.)"""
        key = self._make_key(name)
        dest_path = os.path.join(CUSTOM_REFERENCES_DIR, f"{key}_reference.wav")
        shutil.copyfile(source_audio_path, dest_path)
        self._data[key] = {
            "display_name": name,
            "type": "clone",
            "lang_code": "a",
            "reference_path": dest_path,
            "seed": seed,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
        }
        self._save()
        return key

    def delete(self, key: str):
        if key in self._data:
            meta = self._data.pop(key)
            self._save()
            ref_path = meta.get("reference_path")
            if ref_path and os.path.exists(ref_path):
                try:
                    os.remove(ref_path)
                except Exception as e:
                    print(f"Could not remove reference clip: {e}")


# ---------------------------------------------------------------------------
# UI - the "Create New Voice" window
# ---------------------------------------------------------------------------

class VoiceLabWindow(ctk.CTkToplevel):
    def __init__(
        self,
        parent,
        engine,
        chatterbox_bridge,
        registry: VoiceRegistry,
        voices,
        preview_texts,
        sample_rate,
        on_voice_created=None,
        scale=1.0,
    ):
        super().__init__(parent)
        self.engine = engine
        self.chatterbox_bridge = chatterbox_bridge
        self.registry = registry
        self.voices = voices  # list of (display_name, voice_id, lang_code)
        self.preview_texts = preview_texts
        self.sample_rate = sample_rate
        self.on_voice_created = on_voice_created
        self.scale = scale * LAB_SCALE_BOOST

        # Display names are NOT unique across languages (e.g. three
        # "Santa (Male)"), which would silently collapse in a name->id dict.
        # Disambiguate duplicate labels by appending the voice id.
        self._name_to_id = {}
        self._name_to_lang = {}
        for name, vid, lc in self.voices:
            label = name
            if label in self._name_to_id:
                label = f"{name} [{vid}]"
            self._name_to_id[label] = vid
            self._name_to_lang[label] = lc

        self.imported_path = None            # original file chosen by the user
        self.normalized_import_path = None   # clean mono wav we actually use

        # Preview "takes": each is a generated clip you can replay / pick from.
        self._takes = []          # list of dicts (see _register_take)
        self._take_buttons = []   # parallel list of CTkButton
        self._take_counter = 0
        self._active_take_index = None
        self.takes_empty_label = None

        self.title("Create New Voice")
        self.geometry(f"{self._f(660)}x{self._f(780)}")
        self.minsize(self._f(580), self._f(660))
        self.grab_set()

        self._build_layout()

    def _f(self, base_size: int) -> int:
        return int(base_size * self.scale)

    # ------------------------------------------------------------------
    # Last-used import folder (so the file dialog doesn't reset each time)
    # ------------------------------------------------------------------
    def _load_last_import_dir(self) -> str:
        try:
            if os.path.exists(LAST_IMPORT_DIR_FILE):
                with open(LAST_IMPORT_DIR_FILE, "r", encoding="utf-8") as f:
                    d = f.read().strip()
                if d and os.path.isdir(d):
                    return d
        except Exception:
            pass
        # No remembered folder yet - start somewhere easy to navigate.
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        if os.path.isdir(desktop):
            return desktop
        return os.path.expanduser("~")

    def _save_last_import_dir(self, directory: str):
        try:
            os.makedirs(CUSTOM_VOICES_DIR, exist_ok=True)
            with open(LAST_IMPORT_DIR_FILE, "w", encoding="utf-8") as f:
                f.write(directory)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=3, minsize=self._f(300))  # content (mix / clone)
        self.grid_rowconfigure(2, weight=2, minsize=self._f(150))  # takes + log

        # -- Mode toggle ------------------------------------------------
        mode_row = ctk.CTkFrame(self, fg_color="transparent")
        mode_row.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.mode_var = ctk.StringVar(value="mix")
        mix_radio = ctk.CTkRadioButton(
            mode_row, text="Mix existing voices", variable=self.mode_var,
            value="mix", command=self._on_mode_changed,
            font=ctk.CTkFont(size=self._f(15)),
            radiobutton_width=self._f(20), radiobutton_height=self._f(20),
        )
        mix_radio.pack(side="left", padx=(0, 28))
        clone_radio = ctk.CTkRadioButton(
            mode_row, text="Clone from recording", variable=self.mode_var,
            value="clone", command=self._on_mode_changed,
            font=ctk.CTkFont(size=self._f(15)),
            radiobutton_width=self._f(20), radiobutton_height=self._f(20),
        )
        clone_radio.pack(side="left")

        help_button = ctk.CTkButton(
            mode_row, text="? Help", width=self._f(70), height=self._f(30),
            font=ctk.CTkFont(size=self._f(12)),
            fg_color="gray30", hover_color="gray20",
            command=self._show_help_dialog,
        )
        help_button.pack(side="right")

        # -- Content area (mix / clone panels stacked, one shown at a time) --
        self.content_frame = ctk.CTkFrame(self, corner_radius=12)
        self.content_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

        self._build_mix_panel()
        self._build_clone_panel()

        # -- Takes + log ------------------------------------------------
        self._build_takes_panel()

        # -- Name + actions -----------------------------------------------
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.grid(row=3, column=0, padx=20, pady=(6, 20), sticky="ew")
        bottom.grid_columnconfigure(0, weight=1)

        sample_label = ctk.CTkLabel(
            bottom, text="Sample text (what previews say):",
            font=ctk.CTkFont(size=self._f(13)),
        )
        sample_label.grid(row=0, column=0, sticky="w")

        self.sample_text_entry = ctk.CTkEntry(
            bottom, font=ctk.CTkFont(size=self._f(13)), height=self._f(36),
        )
        self.sample_text_entry.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        self.sample_text_entry.insert(0, self.preview_texts.get("a", ""))

        name_label = ctk.CTkLabel(
            bottom, text="New voice name:", font=ctk.CTkFont(size=self._f(15))
        )
        name_label.grid(row=2, column=0, sticky="w")

        self.name_entry = ctk.CTkEntry(
            bottom, placeholder_text="e.g. Deep Narrator",
            font=ctk.CTkFont(size=self._f(16)), height=self._f(42),
        )
        self.name_entry.grid(row=3, column=0, sticky="ew", pady=(4, 10))

        self.status_label = ctk.CTkLabel(
            bottom, text="", font=ctk.CTkFont(size=self._f(13)),
            wraplength=self._f(560), justify="left", text_color="gray70",
        )
        self.status_label.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        action_row = ctk.CTkFrame(bottom, fg_color="transparent")
        action_row.grid(row=5, column=0, sticky="ew")
        action_row.grid_columnconfigure(0, weight=1)
        action_row.grid_columnconfigure(1, weight=1)

        self.preview_button = ctk.CTkButton(
            action_row, text="▶ Preview Mix", command=self._on_preview_clicked,
            height=self._f(46), font=ctk.CTkFont(size=self._f(15)),
        )
        self.preview_button.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.save_button = ctk.CTkButton(
            action_row, text="💾 Save Voice", command=self._on_save_clicked,
            height=self._f(46), font=ctk.CTkFont(size=self._f(15), weight="bold"),
        )
        self.save_button.grid(row=0, column=1, sticky="ew")

        # Now that preview_button exists, it's safe to sync the visible panel
        # and button labels to the current mode.
        self._on_mode_changed()

        # Start the clone sliders on the default preset's values.
        self._apply_clone_preset("Standard clone")

    # -- Mix panel --------------------------------------------------------
    def _build_mix_panel(self):
        self.mix_panel = ctk.CTkScrollableFrame(self.content_frame, fg_color="transparent")
        # Gridded here as the initial visible panel; _on_mode_changed() swaps
        # this and clone_panel in/out with grid()/grid_remove() afterwards.
        self.mix_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.mix_panel.grid_columnconfigure(0, weight=1)
        self.mix_panel.grid_columnconfigure(1, weight=1)

        info = ctk.CTkLabel(
            self.mix_panel,
            text="Pick two bundled voices to blend into a new one. Use \u25b6 Hear "
                 "to listen to either source voice before you mix them.",
            font=ctk.CTkFont(size=self._f(13)),
            wraplength=self._f(520), justify="left", text_color="gray70",
        )
        info.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        voice_names = list(self._name_to_id.keys())

        label_a = ctk.CTkLabel(self.mix_panel, text="Voice A", font=ctk.CTkFont(size=self._f(14)))
        label_a.grid(row=1, column=0, sticky="w")
        label_b = ctk.CTkLabel(self.mix_panel, text="Voice B", font=ctk.CTkFont(size=self._f(14)))
        label_b.grid(row=1, column=1, sticky="w")

        # -- Voice A: dropdown + hear button side by side --------------------
        row_a = ctk.CTkFrame(self.mix_panel, fg_color="transparent")
        row_a.grid(row=2, column=0, sticky="ew", padx=(0, 10))
        row_a.grid_columnconfigure(0, weight=1)

        self.mix_voice_a = ctk.CTkOptionMenu(
            row_a, values=voice_names, font=ctk.CTkFont(size=self._f(13)),
            dropdown_font=ctk.CTkFont(size=self._f(13)), height=self._f(38),
        )
        self.mix_voice_a.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        if voice_names:
            self.mix_voice_a.set(voice_names[0])

        self.hear_a_button = ctk.CTkButton(
            row_a, text="\u25b6 Hear", width=self._f(72), height=self._f(38),
            font=ctk.CTkFont(size=self._f(13)), fg_color="gray30", hover_color="gray20",
            command=lambda: self._on_hear_bundled_clicked(self.mix_voice_a, self.hear_a_button),
        )
        self.hear_a_button.grid(row=0, column=1)

        # -- Voice B: dropdown + hear button side by side --------------------
        row_b = ctk.CTkFrame(self.mix_panel, fg_color="transparent")
        row_b.grid(row=2, column=1, sticky="ew")
        row_b.grid_columnconfigure(0, weight=1)

        self.mix_voice_b = ctk.CTkOptionMenu(
            row_b, values=voice_names, font=ctk.CTkFont(size=self._f(13)),
            dropdown_font=ctk.CTkFont(size=self._f(13)), height=self._f(38),
        )
        self.mix_voice_b.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        if len(voice_names) > 1:
            self.mix_voice_b.set(voice_names[1])
        elif voice_names:
            self.mix_voice_b.set(voice_names[0])

        self.hear_b_button = ctk.CTkButton(
            row_b, text="\u25b6 Hear", width=self._f(72), height=self._f(38),
            font=ctk.CTkFont(size=self._f(13)), fg_color="gray30", hover_color="gray20",
            command=lambda: self._on_hear_bundled_clicked(self.mix_voice_b, self.hear_b_button),
        )
        self.hear_b_button.grid(row=0, column=1)

        balance_label = ctk.CTkLabel(
            self.mix_panel, text="Balance (A \u2194 B):", font=ctk.CTkFont(size=self._f(14)),
        )
        balance_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(22, 0))

        self.mix_balance_slider = ctk.CTkSlider(
            self.mix_panel, from_=0.0, to=1.0, number_of_steps=20, height=self._f(18),
        )
        self.mix_balance_slider.set(0.5)
        self.mix_balance_slider.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self.mix_balance_value_label = ctk.CTkLabel(
            self.mix_panel, text="Even blend (50 / 50)", font=ctk.CTkFont(size=self._f(13)),
            text_color="gray60",
        )
        self.mix_balance_value_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.mix_balance_slider.configure(command=self._on_balance_changed)

        note = ctk.CTkLabel(
            self.mix_panel,
            text=(
                "Both voices should be from the same language for a clean blend. "
                "Mixing uses Kokoro and is deterministic - the same blend + text "
                "always sounds the same, so there's no seed to set here."
            ),
            wraplength=self._f(520), justify="left", font=ctk.CTkFont(size=self._f(12)),
            text_color="orange",
        )
        note.grid(row=6, column=0, columnspan=2, sticky="w", pady=(20, 0))

    def _on_balance_changed(self, value):
        balance = float(value)
        pct_b = round(balance * 100)
        pct_a = 100 - pct_b
        if pct_a == pct_b:
            self.mix_balance_value_label.configure(text="Even blend (50 / 50)")
        else:
            self.mix_balance_value_label.configure(text=f"{pct_a} / {pct_b}")

    # -- Clone panel --------------------------------------------------------
    def _build_clone_panel(self):
        self.clone_panel = ctk.CTkScrollableFrame(self.content_frame, fg_color="transparent")
        # Built after mix_panel and would otherwise sit visually on top of it.
        # _on_mode_changed() immediately below in _build_layout() calls
        # grid_remove() on whichever panel isn't active, so only one panel is
        # ever actually placed in the grid at a time.
        self.clone_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.clone_panel.grid_columnconfigure(0, weight=1)

        info = ctk.CTkLabel(
            self.clone_panel,
            text=(
                "Import a clean recording (yours, or anyone else's with their "
                f"permission). At least {MIN_CLONE_SECONDS:.0f} seconds with minimal "
                "background noise works best. mp3, m4a, flac, ogg and wav are all "
                "accepted - the clip is converted to a clean wav on import."
            ),
            font=ctk.CTkFont(size=self._f(13)),
            wraplength=self._f(520), justify="left", text_color="gray70",
        )
        info.grid(row=0, column=0, sticky="w", pady=(0, 14))

        import_row = ctk.CTkFrame(self.clone_panel, fg_color="transparent")
        import_row.grid(row=1, column=0, sticky="w")

        import_btn = ctk.CTkButton(
            import_row, text="\U0001F4C1 Import Recording...",
            command=self._on_import_clicked, height=self._f(42),
            font=ctk.CTkFont(size=self._f(14)),
        )
        import_btn.pack(side="left", padx=(0, 10))

        self.hear_imported_button = ctk.CTkButton(
            import_row, text="\u25b6 Hear Recording", width=self._f(150), height=self._f(42),
            font=ctk.CTkFont(size=self._f(14)), fg_color="gray30", hover_color="gray20",
            command=self._on_hear_imported_clicked,
        )
        self.hear_imported_button.pack(side="left")

        self.imported_file_label = ctk.CTkLabel(
            self.clone_panel, text="No file selected.",
            font=ctk.CTkFont(size=self._f(13)), text_color="gray60",
        )
        self.imported_file_label.grid(row=2, column=0, sticky="w", pady=(10, 0))

        # -- Generation controls (these drive the per-take variation) -------
        params = ctk.CTkFrame(self.clone_panel, corner_radius=10)
        params.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        params.grid_columnconfigure(1, weight=1)

        params_title = ctk.CTkLabel(
            params, text="Preview controls (Chatterbox)",
            font=ctk.CTkFont(size=self._f(13), weight="bold"),
        )
        params_title.grid(row=0, column=0, columnspan=3, padx=12, pady=(10, 6), sticky="w")

        # -- Preset / style dropdown -----------------------------------
        preset_label = ctk.CTkLabel(
            params, text="Style preset:", font=ctk.CTkFont(size=self._f(13)),
        )
        preset_label.grid(row=1, column=0, padx=(12, 8), sticky="w")

        self.clone_preset_menu = ctk.CTkOptionMenu(
            params, values=list(CLONE_PRESETS.keys()),
            command=self._apply_clone_preset,
            font=ctk.CTkFont(size=self._f(13)),
            dropdown_font=ctk.CTkFont(size=self._f(13)), height=self._f(34),
        )
        self.clone_preset_menu.grid(row=1, column=1, columnspan=2, sticky="ew",
                                    padx=(0, 12), pady=(0, 2))
        self.clone_preset_menu.set("Standard clone")

        preset_note = ctk.CTkLabel(
            params,
            text="Presets tune delivery only. Accent comes from your recording - "
                 "for an Indian/British/US accent, import a clip in that accent.",
            font=ctk.CTkFont(size=self._f(11)), text_color="gray60",
            wraplength=self._f(480), justify="left",
        )
        preset_note.grid(row=2, column=0, columnspan=3, padx=12, pady=(0, 6), sticky="w")

        # Exaggeration
        exag_label = ctk.CTkLabel(
            params, text="Exaggeration:", font=ctk.CTkFont(size=self._f(13)),
        )
        exag_label.grid(row=3, column=0, padx=(12, 8), sticky="w")

        self.clone_exag_slider = ctk.CTkSlider(params, from_=0.1, to=1.5, number_of_steps=14)
        self.clone_exag_slider.set(0.5)
        self.clone_exag_slider.grid(row=3, column=1, sticky="ew", padx=(0, 8))

        self.clone_exag_value = ctk.CTkLabel(
            params, text="0.50", font=ctk.CTkFont(size=self._f(13)), width=self._f(44),
        )
        self.clone_exag_value.grid(row=3, column=2, padx=(0, 12), sticky="e")
        self.clone_exag_slider.configure(
            command=lambda v: self._on_knob_moved(self.clone_exag_value, v)
        )

        # CFG / pacing - lower = looser & more expressive (lets exaggeration show)
        cfg_label = ctk.CTkLabel(
            params, text="CFG / pacing:", font=ctk.CTkFont(size=self._f(13)),
        )
        cfg_label.grid(row=4, column=0, padx=(12, 8), pady=(8, 0), sticky="w")

        self.clone_cfg_slider = ctk.CTkSlider(params, from_=0.0, to=1.0, number_of_steps=20)
        self.clone_cfg_slider.set(0.5)
        self.clone_cfg_slider.grid(row=4, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))

        self.clone_cfg_value = ctk.CTkLabel(
            params, text="0.50", font=ctk.CTkFont(size=self._f(13)), width=self._f(44),
        )
        self.clone_cfg_value.grid(row=4, column=2, padx=(0, 12), pady=(8, 0), sticky="e")
        self.clone_cfg_slider.configure(
            command=lambda v: self._on_knob_moved(self.clone_cfg_value, v)
        )

        # Temperature - higher = more variety between takes
        temp_label = ctk.CTkLabel(
            params, text="Temperature:", font=ctk.CTkFont(size=self._f(13)),
        )
        temp_label.grid(row=5, column=0, padx=(12, 8), pady=(8, 12), sticky="w")

        self.clone_temp_slider = ctk.CTkSlider(params, from_=0.1, to=1.2, number_of_steps=22)
        self.clone_temp_slider.set(0.8)
        self.clone_temp_slider.grid(row=5, column=1, sticky="ew", padx=(0, 8), pady=(8, 12))

        self.clone_temp_value = ctk.CTkLabel(
            params, text="0.80", font=ctk.CTkFont(size=self._f(13)), width=self._f(44),
        )
        self.clone_temp_value.grid(row=5, column=2, padx=(0, 12), pady=(8, 12), sticky="e")
        self.clone_temp_slider.configure(
            command=lambda v: self._on_knob_moved(self.clone_temp_value, v)
        )

        # -- Seed (fix it for reproducible takes, or leave random) ----------
        seed_label = ctk.CTkLabel(
            params, text="Seed:", font=ctk.CTkFont(size=self._f(13)),
        )
        seed_label.grid(row=6, column=0, padx=(12, 8), pady=(0, 12), sticky="w")

        seed_row = ctk.CTkFrame(params, fg_color="transparent")
        seed_row.grid(row=6, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=(0, 12))
        seed_row.grid_columnconfigure(1, weight=1)

        self.clone_seed_fixed = ctk.BooleanVar(value=False)
        self.clone_seed_check = ctk.CTkCheckBox(
            seed_row, text="Fix", variable=self.clone_seed_fixed,
            command=self._on_seed_fixed_toggled,
            font=ctk.CTkFont(size=self._f(12)),
            checkbox_width=self._f(18), checkbox_height=self._f(18),
        )
        self.clone_seed_check.grid(row=0, column=0, padx=(0, 8))

        self.clone_seed_entry = ctk.CTkEntry(
            seed_row, font=ctk.CTkFont(size=self._f(13)), height=self._f(34),
            state="disabled",
        )
        self.clone_seed_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.clone_seed_entry.configure(state="normal")
        self.clone_seed_entry.insert(0, str(self._new_seed()))
        self.clone_seed_entry.configure(state="disabled")

        self.clone_seed_random_btn = ctk.CTkButton(
            seed_row, text="\U0001F3B2", width=self._f(40), height=self._f(34),
            font=ctk.CTkFont(size=self._f(14)), fg_color="gray30", hover_color="gray20",
            command=self._randomize_seed_entry,
        )
        self.clone_seed_random_btn.grid(row=0, column=2)

        seed_note = ctk.CTkLabel(
            params,
            text="Off = a new random take every Preview (shown here after each "
                 "generation). Check Fix and enter a number to reuse the exact "
                 "same take on your next Preview.",
            font=ctk.CTkFont(size=self._f(11)), text_color="gray60",
            wraplength=self._f(480), justify="left",
        )
        seed_note.grid(row=7, column=0, columnspan=3, padx=12, pady=(0, 10), sticky="w")

    def _on_knob_moved(self, value_label, v):
        """Update a slider's value readout and mark the preset as Custom, since
        the user has hand-tweaked away from a preset."""
        value_label.configure(text=f"{float(v):.2f}")
        try:
            if self.clone_preset_menu.get() != "Custom":
                self.clone_preset_menu.set("Custom")
        except Exception:
            pass

    def _apply_clone_preset(self, name):
        preset = CLONE_PRESETS.get(name)
        if not preset:
            return
        exag, cfg, temp = preset["exaggeration"], preset["cfg_weight"], preset["temperature"]
        self.clone_exag_slider.set(exag)
        self.clone_exag_value.configure(text=f"{exag:.2f}")
        self.clone_cfg_slider.set(cfg)
        self.clone_cfg_value.configure(text=f"{cfg:.2f}")
        self.clone_temp_slider.set(temp)
        self.clone_temp_value.configure(text=f"{temp:.2f}")
        # Re-assert the preset name (the slider .set() calls above don't fire
        # the user-move handler, so this stays showing the chosen preset).
        self.clone_preset_menu.set(name)

    def _new_seed(self):
        """A fresh random seed. Used whenever the seed isn't fixed."""
        return random.randint(0, SEED_MAX)

    # -- Seed field helpers --------------------------------------------
    def _on_seed_fixed_toggled(self):
        if self.clone_seed_fixed.get():
            self.clone_seed_entry.configure(state="normal")
        else:
            self.clone_seed_entry.configure(state="disabled")

    def _randomize_seed_entry(self):
        """Rolls a new seed into the field, regardless of the Fix checkbox."""
        was_disabled = self.clone_seed_entry.cget("state") == "disabled"
        self.clone_seed_entry.configure(state="normal")
        self.clone_seed_entry.delete(0, "end")
        self.clone_seed_entry.insert(0, str(self._new_seed()))
        if was_disabled and not self.clone_seed_fixed.get():
            self.clone_seed_entry.configure(state="disabled")

    def _display_used_seed(self, seed):
        """Writes the seed that was actually used for the last take into the
        field, so an unfixed seed is still visible (and easy to then fix)."""
        was_disabled = self.clone_seed_entry.cget("state") == "disabled"
        self.clone_seed_entry.configure(state="normal")
        self.clone_seed_entry.delete(0, "end")
        self.clone_seed_entry.insert(0, str(seed))
        if was_disabled and not self.clone_seed_fixed.get():
            self.clone_seed_entry.configure(state="disabled")

    def _get_clone_seed(self):
        """Returns the seed for the next clone generation: the user's fixed
        value if 'Fix' is checked (validated), otherwise a fresh random one."""
        if self.clone_seed_fixed.get():
            raw = self.clone_seed_entry.get().strip()
            try:
                seed = int(raw)
                if not (0 <= seed <= SEED_MAX):
                    raise ValueError
                return seed
            except ValueError:
                self._set_status(
                    f"Seed must be a whole number 0-{SEED_MAX}. Using a random seed instead.",
                    error=True,
                )
                return self._new_seed()
        return self._new_seed()

    def _sample_text(self):
        """The line previews should speak - user-editable, with a fallback."""
        try:
            txt = self.sample_text_entry.get().strip()
        except Exception:
            txt = ""
        return txt or self.preview_texts.get("a", "This is a preview of how I sound.")

    # -- Takes + log panel -------------------------------------------------
    def _build_takes_panel(self):
        frame = ctk.CTkFrame(self, corner_radius=12)
        frame.grid(row=2, column=0, padx=20, pady=(0, 8), sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        header = ctk.CTkLabel(
            frame,
            text="Preview takes  -  click one to replay it and pick it for saving",
            font=ctk.CTkFont(size=self._f(13), weight="bold"),
        )
        header.grid(row=0, column=0, padx=14, pady=(12, 4), sticky="w")

        self.takes_list = ctk.CTkScrollableFrame(frame, corner_radius=8, height=self._f(120))
        self.takes_list.grid(row=1, column=0, padx=14, pady=(0, 8), sticky="nsew")
        self.takes_list.grid_columnconfigure(0, weight=1)

        self.takes_empty_label = ctk.CTkLabel(
            self.takes_list,
            text="No takes yet - hit Preview to generate one.",
            font=ctk.CTkFont(size=self._f(12)), text_color="gray60",
        )
        self.takes_empty_label.grid(row=0, column=0, padx=8, pady=8, sticky="w")

        log_label = ctk.CTkLabel(
            frame, text="Log", font=ctk.CTkFont(size=self._f(12)), text_color="gray60",
        )
        log_label.grid(row=2, column=0, padx=14, sticky="w")

        self.log_box = ctk.CTkTextbox(
            frame, height=self._f(90), font=ctk.CTkFont(size=self._f(12)), wrap="word",
        )
        self.log_box.grid(row=3, column=0, padx=14, pady=(2, 12), sticky="ew")
        self.log_box.configure(state="disabled")

    def _log(self, message: str):
        def _append():
            stamp = time.strftime("%H:%M:%S")
            self.log_box.configure(state="normal")
            self.log_box.insert("end", f"[{stamp}] {message}\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(0, _append)

    def _show_help_dialog(self):
        win = ctk.CTkToplevel(self)
        win.title("Create New Voice - Help")
        win.geometry(f"{self._f(540)}x{self._f(620)}")
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
                anchor="w", justify="left", wraplength=self._f(470),
            )
            b.pack(fill="x", pady=(0, 4))

        add_section(
            "Two ways to make a voice",
            "Mix - blend two of the bundled voices into a new one. This uses "
            "Kokoro and is deterministic: the same blend always sounds the "
            "same.\n\n"
            "Clone - import a real recording and speak new text in that voice. "
            "This uses Chatterbox, which is slower and slightly different every "
            "run unless you fix the seed (see below)."
        )
        add_section(
            "Mixing - Voice A / B and Balance",
            "Pick two source voices, then use the Balance slider to set how "
            "much of each goes into the blend (e.g. 70/30). The \u25b6 Hear "
            "buttons play each source on its own so you can choose before "
            "blending. Keep both voices in the same language for a clean "
            "result."
        )
        add_section(
            "Cloning - Import Recording",
            "Choose a clean clip of the target voice (yours, or someone else's "
            f"with permission). At least {MIN_CLONE_SECONDS:.0f} seconds with "
            "little background noise works best. mp3, m4a, flac, ogg and wav "
            "are all accepted - the clip is converted to a clean mono wav "
            "automatically. Use \u25b6 Hear Recording to confirm it imported "
            "correctly."
        )
        add_section(
            "Accent - important",
            "For a CLONE, the accent, timbre and speaking style all come from "
            "the recording you import - the model copies whatever it hears. "
            "There is no 'accent' setting, and the sliders below do NOT change "
            "accent. Want an Indian (or any) accent? Import a clip of someone "
            "speaking with that accent.\n\n"
            "For a MIX, the accent is fixed by the bundled voice + language you "
            "pick (US English, UK English, Hindi, etc.) - you're limited to "
            "what's shipped, and can't dial an arbitrary accent."
        )
        add_section(
            "Style preset (dropdown)",
            "A quick way to set the three sliders below to sensible values:\n"
            "\u2022 Faithful (1:1 clone) - sticks closest to your recording; "
            "best for preserving the accent.\n"
            "\u2022 Soft clone - gentle and natural.\n"
            "\u2022 Standard clone - balanced default.\n"
            "\u2022 Expressive - livelier, more dramatic.\n"
            "\u2022 US / British / Indian English - these only bias the "
            "delivery toward faithful reproduction; they do NOT add an accent "
            "on their own. The accent still comes from your clip.\n\n"
            "Moving any slider by hand switches the preset to 'Custom'."
        )
        add_section(
            "Sample text",
            "The line previews speak. Edit it to test the voice on wording you "
            "actually care about - it doesn't affect the saved voice, only what "
            "you hear while auditioning takes."
        )
        add_section(
            "Exaggeration",
            "How strongly Chatterbox leans into the expressive character of the "
            "delivery (clones only; mixing ignores it). It mostly shows when "
            "CFG is lowish - at high CFG the effect is muted, which is why it "
            "can seem to 'do nothing'. Try exaggeration ~0.7 with CFG ~0.3 for "
            "a livelier read; keep both near 0.5 for neutral."
        )
        add_section(
            "CFG / pacing",
            "Guidance strength. Lower (~0.3) = looser, more expressive and lets "
            "exaggeration come through; higher (~0.7) = tighter and flatter, "
            "closer to a plain read. This is the knob that makes exaggeration "
            "actually audible."
        )
        add_section(
            "Temperature",
            "Sampling randomness. Higher = more variety and surprise between "
            "takes; lower = safer and more consistent. Around 0.8 is a good "
            "starting point."
        )
        add_section(
            "Seed",
            "Controls exactly which random 'take' Chatterbox generates. Leave "
            "'Fix' unchecked and every Preview rolls a new seed (shown in the "
            "field afterwards, and in the log/take list). Check 'Fix' and type "
            "a number to force every future Preview to reproduce that exact "
            "take - handy once you've found one you like and want to tweak "
            "wording without the delivery drifting. The \U0001F3B2 button rolls "
            "a fresh number into the field any time."
        )
        add_section(
            "Preview takes + Log",
            "Each preview is added to the Takes list along with its settings "
            "and seed. Click any take to replay it and select it - its sliders "
            "and seed load back in, and when you Save that take's exact "
            "settings (including its seed) are pinned onto the voice, so it "
            "sounds the same every time you generate with it later. The Log "
            "records imports, generations and errors."
        )
        add_section(
            "Saving",
            "Give the voice a name and click Save Voice. It then appears under "
            "'\u2605 My Voices' in the main window. For a clone, pick the take "
            "you like first so its exact settings get saved."
        )

        close_btn = ctk.CTkButton(win, text="Close", command=win.destroy)
        close_btn.pack(pady=(0, 16))

    def _on_mode_changed(self):
        """Show whichever of mix_panel / clone_panel matches the current
        mode and hide the other. Both panels live in the same grid cell
        (row=0, column=0) inside content_frame; using grid()/grid_remove()
        here - rather than tkraise() - is the reliable way to swap them,
        since CTkScrollableFrame's internal canvas doesn't always restack
        correctly with tkraise() alone."""
        if self.mode_var.get() == "mix":
            self.clone_panel.grid_remove()
            self.mix_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
            self.preview_button.configure(text="▶ Preview Mix")
        else:
            self.mix_panel.grid_remove()
            self.clone_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
            self.preview_button.configure(text="▶ Preview Clone")

    # ------------------------------------------------------------------
    # Import + normalisation
    # ------------------------------------------------------------------
    def _normalize_to_wav(self, src_path: str):
        """Decode any supported audio file into a clean mono WAV we can
        reliably preview AND hand to Chatterbox. Returns (wav_path, duration)
        or raises RuntimeError with a helpful message."""
        import numpy as np

        data = None
        sr = None

        # 1) soundfile handles wav/flac/ogg always, and mp3/m4a on newer
        #    libsndfile builds.
        try:
            data, sr = sf.read(src_path, dtype="float32", always_2d=True)
        except Exception:
            data = None

        # 2) Fall back to ffmpeg (handles mp3/m4a/etc everywhere it's present).
        if data is None:
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", src_path,
                     "-ac", "1", "-ar", str(NORMALIZED_SR), tmp_wav],
                    capture_output=True, check=True,
                )
                data, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
            except FileNotFoundError:
                raise RuntimeError(
                    "Couldn't decode that file. Install ffmpeg, or convert the "
                    "clip to WAV/FLAC and import that instead."
                )
            except subprocess.CalledProcessError:
                raise RuntimeError("ffmpeg couldn't decode that file.")
            finally:
                if os.path.exists(tmp_wav) and data is None:
                    try:
                        os.remove(tmp_wav)
                    except Exception:
                        pass

        # Downmix to mono.
        if data.ndim == 2 and data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data.reshape(-1)

        out_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(out_wav, data, sr)
        duration = len(data) / float(sr) if sr else 0.0
        return out_wav, duration

    def _on_import_clicked(self):
        path = filedialog.askopenfilename(
            title="Choose a voice recording",
            initialdir=self._load_last_import_dir(),
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self._save_last_import_dir(os.path.dirname(path))
        self.status_label.configure(text="Loading recording...", text_color="gray70")
        self.update_idletasks()

        try:
            wav_path, duration = self._normalize_to_wav(path)
        except Exception as e:
            self.imported_path = None
            self.normalized_import_path = None
            self.imported_file_label.configure(
                text="No file selected.", text_color="gray60"
            )
            self.status_label.configure(text=str(e), text_color="#E06C6C")
            self._log(f"Import failed: {e}")
            return

        self.imported_path = path
        self.normalized_import_path = wav_path
        self._log(f"Imported {os.path.basename(path)} ({duration:.1f}s).")

        if duration < MIN_CLONE_SECONDS:
            self.status_label.configure(
                text=(
                    f"That clip is only {duration:.1f}s long - at least "
                    f"{MIN_CLONE_SECONDS:.0f}s is recommended for a good clone. "
                    "You can still save it."
                ),
                text_color="#E0A64C",
            )
        else:
            self.status_label.configure(text="Recording ready.", text_color="gray70")

        self.imported_file_label.configure(
            text=f"{os.path.basename(path)} ({duration:.1f}s)", text_color="gray90",
        )

    # ------------------------------------------------------------------
    # Hearing source voices before you commit to a mix / clone
    # ------------------------------------------------------------------
    def _on_hear_bundled_clicked(self, option_menu, button):
        """Plays back one of the two bundled voices selected for mixing,
        so you can hear each source before deciding on a blend."""
        label = option_menu.get()
        voice_id = self._name_to_id.get(label)
        lang_code = self._name_to_lang.get(label, "a")
        if not voice_id:
            return

        button.configure(state="disabled")
        self.status_label.configure(text=f"Loading '{label}'...", text_color="gray70")

        def worker():
            try:
                os.makedirs(CHARACTER_SPEECH_DIR, exist_ok=True)
                sample_path = os.path.join(CHARACTER_SPEECH_DIR, f"{voice_id}.wav")
                if not os.path.exists(sample_path):
                    text = self.preview_texts.get(lang_code, self.preview_texts.get("a"))
                    audio = self.engine.synthesize(text, voice_id, lang_code)
                    sf.write(sample_path, audio, self.sample_rate)
                self._play_audio_file(sample_path)
                self._set_status(f"Playing '{label}'.")
            except Exception as e:
                self._set_status(f"Error: {e}", error=True)
                self._log(f"Hear failed: {e}")
            finally:
                self.after(0, lambda: button.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_hear_imported_clicked(self):
        target = self.normalized_import_path or self.imported_path
        if not target:
            self.status_label.configure(
                text="Import a recording first.", text_color="#E0A64C"
            )
            return
        self._play_audio_file(target)
        self._set_status("Playing your imported recording.")

    # ------------------------------------------------------------------
    # Preview mix / clone result -> becomes a "take"
    # ------------------------------------------------------------------
    def _current_mix_voice_string(self):
        a_label = self.mix_voice_a.get()
        b_label = self.mix_voice_b.get()
        a_id = self._name_to_id[a_label]
        b_id = self._name_to_id[b_label]
        lang_code = self._name_to_lang[a_label]

        balance = float(self.mix_balance_slider.get())
        weight_a = round(1.0 - balance, 2)
        weight_b = round(balance, 2)
        mix_voice_id = f"{a_id}({weight_a}),{b_id}({weight_b})"
        return mix_voice_id, lang_code, a_label, b_label, weight_a, weight_b

    def _on_preview_clicked(self):
        mode = self.mode_var.get()

        # Resolve parameters on the main thread so the UI stays consistent.
        if mode == "clone":
            if not self.normalized_import_path:
                self._set_status("Import a recording first.", error=True)
                return
            seed = self._get_clone_seed()
            self._display_used_seed(seed)
            exaggeration = float(self.clone_exag_slider.get())
            cfg_weight = float(self.clone_cfg_slider.get())
            temperature = float(self.clone_temp_slider.get())
        else:
            mix_voice_id, lang_code, a_label, b_label, wa, wb = self._current_mix_voice_string()

        sample_text = self._sample_text()
        self.preview_button.configure(state="disabled")
        self.status_label.configure(text="Generating preview...", text_color="gray70")

        def worker():
            try:
                if mode == "clone":
                    self._log(
                        f"Generating clone take - seed {seed}, exag {exaggeration:.2f}, "
                        f"cfg {cfg_weight:.2f}, temp {temperature:.2f}..."
                    )
                    audio = self.chatterbox_bridge.synthesize_chunk_with_reference(
                        sample_text,
                        self.normalized_import_path,
                        exaggeration=exaggeration,
                        seed=seed,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                    )
                    self._persist_take(
                        audio, mode="clone",
                        params={
                            "seed": seed, "exaggeration": exaggeration,
                            "cfg_weight": cfg_weight, "temperature": temperature,
                        },
                        detail=(f"seed {seed} \u00b7 exag {exaggeration:.2f} "
                                f"\u00b7 cfg {cfg_weight:.2f} \u00b7 temp {temperature:.2f}"),
                    )
                else:
                    self._log(f"Generating mix take - {a_label} {wa} / {b_label} {wb}...")
                    audio = self.engine.synthesize(sample_text, mix_voice_id, lang_code)
                    self._persist_take(
                        audio, mode="mix", params={},
                        detail=f"{int(wa*100)}/{int(wb*100)} blend",
                    )
                self._set_status("Preview ready - added to takes below.")
            except Exception as e:
                self._set_status(f"Error: {e}", error=True)
                self._log(f"Preview failed: {e}")
            finally:
                self.after(0, lambda: self.preview_button.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _persist_take(self, audio, mode, params, detail):
        """Write the generated audio to a temp wav and register it as a take
        on the main thread (auto-selected + played)."""
        tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(tmp_path, audio, self.sample_rate)
        self.after(0, lambda: self._register_take(tmp_path, mode, params, detail))

    def _register_take(self, wav_path, mode, params, detail):
        if self.takes_empty_label is not None:
            self.takes_empty_label.destroy()
            self.takes_empty_label = None

        self._take_counter += 1
        take = {
            "n": self._take_counter,
            "wav_path": wav_path,
            "mode": mode,
            "params": params or {},
            "detail": detail,
        }
        self._takes.append(take)

        idx = len(self._takes) - 1
        btn = ctk.CTkButton(
            self.takes_list,
            text=f"\u25b6 Take {take['n']}  ({mode})  \u00b7  {detail}",
            anchor="w", height=self._f(34),
            font=ctk.CTkFont(size=self._f(13)),
            fg_color="transparent", text_color=("gray10", "gray90"),
            hover_color=("gray80", "gray30"),
            command=lambda i=idx: self._select_take(i),
        )
        btn.grid(row=idx, column=0, padx=4, pady=2, sticky="ew")
        self._take_buttons.append(btn)

        # Trim old takes so the list doesn't grow forever.
        while len(self._takes) > MAX_TAKES:
            old = self._takes.pop(0)
            old_btn = self._take_buttons.pop(0)
            old_btn.destroy()
            try:
                if os.path.exists(old["wav_path"]):
                    os.remove(old["wav_path"])
            except Exception:
                pass

        # Re-bind commands + rows so indices stay correct after any trim.
        for new_i, b in enumerate(self._take_buttons):
            b.configure(command=lambda i=new_i: self._select_take(i))
            b.grid_configure(row=new_i)

        self._log(f"Take {take['n']} ready ({detail}).")
        self._select_take(len(self._takes) - 1)

    def _select_take(self, index):
        if index < 0 or index >= len(self._takes):
            return
        self._active_take_index = index
        take = self._takes[index]

        for i, b in enumerate(self._take_buttons):
            b.configure(fg_color=("gray75", "gray25") if i == index else "transparent")

        # A clone take carries its full delivery settings - load them into the
        # controls so Save pins exactly what you're hearing. (Its seed is also
        # written into the seed field, so re-checking Fix pins it going forward.)
        if take["mode"] == "clone":
            p = take.get("params", {})
            if p.get("exaggeration") is not None:
                self.clone_exag_slider.set(p["exaggeration"])
                self.clone_exag_value.configure(text=f"{p['exaggeration']:.2f}")
            if p.get("cfg_weight") is not None:
                self.clone_cfg_slider.set(p["cfg_weight"])
                self.clone_cfg_value.configure(text=f"{p['cfg_weight']:.2f}")
            if p.get("temperature") is not None:
                self.clone_temp_slider.set(p["temperature"])
                self.clone_temp_value.configure(text=f"{p['temperature']:.2f}")
            if p.get("seed") is not None:
                self._display_used_seed(p["seed"])

        self._play_audio_file(take["wav_path"])
        self._set_status(
            f"Playing Take {take['n']} ({take['detail']}). It's now selected for saving."
        )

    def _active_clone_take(self):
        if self._active_take_index is None:
            return None
        take = self._takes[self._active_take_index]
        return take if take["mode"] == "clone" else None

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _on_save_clicked(self):
        name = self.name_entry.get().strip()
        if not name:
            self.status_label.configure(
                text="Give the new voice a name first.", text_color="#E0A64C"
            )
            return

        if self.mode_var.get() == "mix":
            mix_voice_id, lang_code, *_ = self._current_mix_voice_string()
            self.registry.add_mix_voice(name, mix_voice_id, lang_code)
            self._set_status(f"Saved '{name}' as a mixed voice.")
            self._log(f"Saved mix voice '{name}'.")
        else:
            if not self.normalized_import_path:
                self.status_label.configure(
                    text="Import a recording first.", text_color="#E0A64C"
                )
                return
            # Prefer the take you actually picked; else fall back to controls.
            take = self._active_clone_take()
            if take is not None:
                p = take.get("params", {})
                seed = p.get("seed")
                exaggeration = p.get("exaggeration", float(self.clone_exag_slider.get()))
                cfg_weight = p.get("cfg_weight", float(self.clone_cfg_slider.get()))
                temperature = p.get("temperature", float(self.clone_temp_slider.get()))
            else:
                # No take picked - use the seed field (fixed value if 'Fix' is
                # checked, otherwise a fresh one) so the voice is still
                # reproducible later, using the current slider values.
                seed = self._get_clone_seed()
                exaggeration = float(self.clone_exag_slider.get())
                cfg_weight = float(self.clone_cfg_slider.get())
                temperature = float(self.clone_temp_slider.get())

            self.registry.add_clone_voice(
                name, self.normalized_import_path,
                seed=seed, exaggeration=exaggeration,
                cfg_weight=cfg_weight, temperature=temperature,
            )
            seed_note = f"seed {seed}" if seed is not None else "random seed each run"
            self._set_status(f"Saved '{name}' as a cloned voice ({seed_note}).")
            self._log(
                f"Saved clone voice '{name}' ({seed_note}, exag {exaggeration:.2f}, "
                f"cfg {cfg_weight:.2f}, temp {temperature:.2f})."
            )

        if self.on_voice_created:
            self.on_voice_created()

    def _set_status(self, text, error=False):
        color = "#E06C6C" if error else "gray70"
        self.after(0, lambda: self.status_label.configure(text=text, text_color=color))

    def _play_audio_file(self, path):
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Playback error: {e}")

    def _play_audio_array(self, audio):
        tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(tmp_path, audio, self.sample_rate)
        self._play_audio_file(tmp_path)