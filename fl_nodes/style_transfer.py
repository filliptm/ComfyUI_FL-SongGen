"""
FL Song Gen Style Transfer Node.
Generate songs using reference audio for style conditioning.
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

SongGenWrapper = _songgen_wrapper.SongGenWrapper
empty_audio = _audio_utils.empty_audio
prepare_prompt_audio_for_songgen = _audio_utils.prepare_prompt_audio_for_songgen


class FL_SongGen_StyleTransfer:
    """
    Generate a song using reference audio for style transfer.

    Provide a reference audio clip (ideally a chorus section) and the model
    will generate new music matching that style with your lyrics.
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
                "reference_audio": (
                    "AUDIO",
                    {
                        "tooltip": "Reference audio for style (max 10 seconds used)"
                    }
                ),
            },
            "optional": {
                "description": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional style description to combine with reference audio (e.g., 'female, pop, emotional')"
                    }
                ),
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
        reference_audio: dict,
        description: str = "",
        duration: float = 60.0,
        temperature: float = 1.0,
        cfg_coef: float = 1.5,
        top_k: int = 50,
        gen_type: str = "mixed",
        seed: int = -1
    ) -> Tuple[dict, dict, dict]:
        """
        Generate song with style transfer from reference audio.

        Args:
            model: Loaded model info dict
            lyrics: Formatted lyrics with section tags
            reference_audio: ComfyUI AUDIO dict for style reference
            description: Optional style description to combine with reference audio
            duration: Target duration in seconds
            temperature: Sampling temperature
            cfg_coef: Classifier-free guidance strength
            top_k: Top-k sampling parameter
            gen_type: Output type (mixed, separate, vocal, bgm)
            seed: Random seed

        Returns:
            (mixed_audio, vocal_audio, bgm_audio) as ComfyUI AUDIO dicts
        """
        print(f"\n{'='*60}")
        print(f"[FL SongGen Style Transfer] Starting Generation")
        print(f"{'='*60}")
        print(f"Duration: {duration}s")
        print(f"Temperature: {temperature}")
        print(f"CFG: {cfg_coef}")
        print(f"Top-K: {top_k}")
        print(f"Gen Type: {gen_type}")
        print(f"Seed: {seed}")
        print(f"Description: {description[:50]}..." if description else "Description: None")
        print(f"Reference Audio: {reference_audio['waveform'].shape}, {reference_audio['sample_rate']}Hz")
        print(f"{'='*60}\n")

        # Validate duration
        max_dur = model.get("max_duration", 150)
        if duration > max_dur:
            print(f"[FL SongGen] WARNING: Duration {duration}s exceeds model max {max_dur}s, clamping.")
            duration = max_dur

        # Prepare reference audio for SongGen (48kHz, max 10s)
        print("[FL SongGen Style Transfer] Preparing reference audio...")
        prompt_audio = prepare_prompt_audio_for_songgen(
            reference_audio,
            max_duration_sec=10.0,
            target_sr=48000
        )

        ref_duration = prompt_audio.shape[-1] / 48000
        print(f"[FL SongGen Style Transfer] Reference audio: {ref_duration:.2f}s at 48kHz")

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
                description=description if description.strip() else None,
                prompt_audio=prompt_audio,
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
            print(f"[FL SongGen Style Transfer] Generation Complete!")
            print(f"Mixed Audio: {mixed_audio['waveform'].shape}, {mixed_audio['sample_rate']}Hz")
            print(f"{'='*60}\n")

            return (mixed_audio, vocal_audio, bgm_audio)

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"[FL SongGen Style Transfer] ERROR: Generation failed!")
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
