"""
FL Song Gen Auto Style Node.
Generate songs using preset style prompts.
"""

import sys
import os
from typing import Tuple
import importlib.util

from comfy.utils import ProgressBar

# Get the package root directory
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))

# Import modules explicitly from our package to avoid conflicts with other FL packages
def _import_from_package(module_name, file_name):
    """Import a module from our package specifically."""
    module_path = os.path.join(_PACKAGE_ROOT, "fl_utils", f"{file_name}.py")
    spec = importlib.util.spec_from_file_location(f"songgen_{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import our modules
_songgen_wrapper = _import_from_package("songgen_wrapper", "songgen_wrapper")
_audio_utils = _import_from_package("audio_utils", "audio_utils")
_model_manager = _import_from_package("model_manager", "model_manager")

SongGenWrapper = _songgen_wrapper.SongGenWrapper
empty_audio = _audio_utils.empty_audio
AUTO_STYLE_PRESETS = _model_manager.AUTO_STYLE_PRESETS


class FL_SongGen_AutoStyle:
    """
    Generate a song using preset auto-style prompts.

    This node uses pre-trained style tokens to condition generation
    without needing reference audio. Choose from genres like Pop, Rock,
    Jazz, etc.
    """

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("mixed_audio", "vocal_audio", "bgm_audio")
    FUNCTION = "generate"
    CATEGORY = "FL Song Gen"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "SONGGEN_MODEL",
                    {
                        "tooltip": "Loaded SongGeneration model"
                    }
                ),
                "lyrics": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[intro-short] ; [verse] Hello world.This is a test ; [chorus] Singing along.Making music ; [outro-short]",
                        "tooltip": "Formatted lyrics with section tags"
                    }
                ),
                "auto_style": (
                    AUTO_STYLE_PRESETS,
                    {
                        "default": "Pop",
                        "tooltip": "Preset style for generation"
                    }
                ),
            },
            "optional": {
                "duration": (
                    "FLOAT",
                    {
                        "default": 60.0,
                        "min": 30.0,
                        "max": 270.0,
                        "step": 5.0,
                        "tooltip": "Target duration in seconds"
                    }
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature. 1.0 = balanced (official default). Lower = more consistent, higher = more creative."
                    }
                ),
                "cfg_coef": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.5,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance. 1.5 = official default. Higher = stronger adherence to lyrics/description."
                    }
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 5000,
                        "step": 1,
                        "tooltip": "Top-k sampling. 50 = focused (official default). Higher values (500-5000) = more diverse/experimental."
                    }
                ),
                "gen_type": (
                    ["mixed", "separate", "vocal", "bgm"],
                    {
                        "default": "mixed",
                        "tooltip": "mixed = combined song. separate = vocal + BGM + mixed tracks. vocal/bgm = single track."
                    }
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483647,
                        "tooltip": "Random seed (-1 for random)"
                    }
                ),
            }
        }


    def generate(
        self,
        model: dict,
        lyrics: str,
        auto_style: str,
        duration: float = 60.0,
        temperature: float = 1.0,
        cfg_coef: float = 1.5,
        top_k: int = 50,
        gen_type: str = "mixed",
        seed: int = -1
    ) -> Tuple[dict, dict, dict]:
        print(f"\n{'='*60}")
        print(f"[FL SongGen Auto Style] Starting Generation")
        print(f"{'='*60}")
        print(f"Style: {auto_style}")
        print(f"Duration: {duration}s")
        print(f"Temperature: {temperature}")
        print(f"CFG: {cfg_coef}")
        print(f"Top-K: {top_k}")
        print(f"Gen Type: {gen_type}")
        print(f"Seed: {seed}")
        print(f"{'='*60}\n")

        # Check if auto prompts are available
        if model.get("auto_prompts") is None:
            print("[FL SongGen Auto Style] WARNING: Auto prompts not loaded!")
            print("Please ensure 'new_prompt.pt' is in ComfyUI/models/songgen/")
            print("Falling back to no style conditioning...")

        # Validate duration
        max_dur = model.get("max_duration", 150)
        if duration > max_dur:
            print(f"[FL SongGen] WARNING: Duration {duration}s exceeds model max {max_dur}s, clamping.")
            duration = max_dur

        # Create wrapper and set up progress bar
        wrapper = SongGenWrapper(model)

        # Progress bar will be created on first callback with actual total
        pbar_holder = [None]
        actual_total = [0]

        def progress_callback(current, total):
            if pbar_holder[0] is None:
                pbar_holder[0] = ProgressBar(total)
                actual_total[0] = total
            pbar_holder[0].update_absolute(current)
            if current % 100 == 0:
                print(f"[FL SongGen] Progress: {current}/{total} tokens")

        wrapper.set_progress_callback(progress_callback)

        # Generate
        try:
            mixed_audio, vocal_audio, bgm_audio = wrapper.generate(
                lyrics=lyrics,
                auto_style=auto_style,
                duration=duration,
                temperature=temperature,
                cfg_coef=cfg_coef,
                top_k=top_k,
                gen_type=gen_type,
                seed=seed,
            )

            if pbar_holder[0] is not None:
                pbar_holder[0].update_absolute(actual_total[0])

            print(f"\n{'='*60}")
            print(f"[FL SongGen Auto Style] Generation Complete!")
            print(f"Mixed Audio: {mixed_audio['waveform'].shape}, {mixed_audio['sample_rate']}Hz")
            print(f"{'='*60}\n")

            return (mixed_audio, vocal_audio, bgm_audio)

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"[FL SongGen Auto Style] ERROR: Generation failed!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")

            sample_rate = model.get("sample_rate", 24000)
            return (
                empty_audio(sample_rate),
                empty_audio(sample_rate),
                empty_audio(sample_rate)
            )
