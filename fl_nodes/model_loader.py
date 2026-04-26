"""
FL Song Gen Model Loader Node.
Loads the SongGeneration model with configurable options.
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

# Import our model_manager module
_model_manager = _import_from_package("model_manager", "model_manager")

load_model = _model_manager.load_model
get_variant_list = _model_manager.get_variant_list
get_variant_info = _model_manager.get_variant_info
MODEL_VARIANTS = _model_manager.MODEL_VARIANTS
clear_model_cache = _model_manager.clear_model_cache
get_recommended_memory_mode = _model_manager.get_recommended_memory_mode
get_available_vram_gb = _model_manager.get_available_vram_gb
set_keep_loaded = _model_manager.set_keep_loaded

# Memory modes
MEMORY_MODES = ["auto", "normal", "low", "ultra_low_mem"]


class FL_SongGen_ModelLoader:
    """
    Load SongGeneration model with variant selection and memory options.

    This node loads the AI song generation model. Different variants offer
    trade-offs between quality, speed, and VRAM requirements.
    """

    RETURN_TYPES = ("SONGGEN_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "FL Song Gen"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution when model cache was cleared (e.g. after Unload Models)
        try:
            import comfy.model_management as cmm
            cache = getattr(cmm, '_songgen_cache', {})
            if not cache:
                return float("nan")  # Always re-run when cache is empty
        except ImportError:
            pass
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        variants = get_variant_list()
        return {
            "required": {
                "model_variant": (
                    variants,
                    {
                        "default": "songgeneration_v2_large",
                        "tooltip": "Model variant. v2-large = best quality, multilingual. base_new = lighter, English+Chinese."
                    }
                ),
            },
            "optional": {
                "memory_mode": (
                    MEMORY_MODES,
                    {
                        "default": "auto",
                        "tooltip": "auto picks the best mode for your GPU. normal = fastest. low = offloads between phases. ultra_low_mem = minimum ~6GB VRAM."
                    }
                ),
                "force_reload": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force reload model even if already cached."
                    }
                ),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Keep model in VRAM between runs. Disable to free VRAM for other models between generations."
                    }
                ),
            }
        }

    def load_model(
        self,
        model_variant: str,
        memory_mode: str = "auto",
        force_reload: bool = False,
        keep_model_loaded: bool = True,
    ) -> Tuple[dict]:
        # Resolve memory mode
        if memory_mode == "auto":
            resolved_mode = get_recommended_memory_mode(model_variant)
            available_vram = get_available_vram_gb()
            print(f"[FL SongGen] Auto-detected memory mode: {resolved_mode} (available VRAM: {available_vram:.1f}GB)")
        else:
            resolved_mode = memory_mode

        # Map mode to flags
        low_mem = resolved_mode in ("low", "ultra_low_mem", "low_mem")
        ultra_low_mem = resolved_mode in ("ultra", "ultra_low_mem")

        # Get variant info for logging
        variant_info = get_variant_info(model_variant)
        print(f"\n{'='*60}")
        print(f"[FL SongGen] Loading Model")
        print(f"{'='*60}")
        print(f"Variant: {model_variant}")
        print(f"Description: {variant_info['description']}")
        print(f"Max Duration: {variant_info['max_duration']}s ({variant_info['max_duration']//60}m {variant_info['max_duration']%60}s)")
        print(f"Languages: {', '.join(variant_info['languages'])}")
        print(f"Memory Mode: {resolved_mode}")
        if ultra_low_mem:
            print(f"VRAM Required: ~6GB (ultra low memory mode)")
        elif low_mem:
            print(f"VRAM Required: {variant_info['vram_low']}GB")
        else:
            print(f"VRAM Required: {variant_info['vram_normal']}GB")
        print(f"{'='*60}\n")

        try:
            # 4 steps for full model loading
            pbar = ProgressBar(4)

            def progress_callback(current, total):
                pbar.update_absolute(current)

            model_info = load_model(
                variant=model_variant,
                low_mem=low_mem or ultra_low_mem,
                use_flash_attn=True,
                force_reload=force_reload,
                progress_callback=progress_callback,
                attention_backend="sdpa",
            )

            # Add ultra_low_mem flag to model_info for generation phase
            model_info["ultra_low_mem"] = ultra_low_mem

            # Configure ComfyUI VRAM integration
            set_keep_loaded(keep_model_loaded)

            print(f"[FL SongGen] Model loaded successfully! (keep_model_loaded={keep_model_loaded})")
            return (model_info,)

        except FileNotFoundError as e:
            print(f"\n{'='*60}")
            print(f"[FL SongGen] ERROR: Model files not found!")
            print(f"{'='*60}")
            print(f"Please download the model from HuggingFace:")
            print(f"https://huggingface.co/aslp-lab/SongGeneration")
            print(f"\nExpected location: ComfyUI/models/songgen/{model_variant}/")
            print(f"Required files: config.yaml, model.pt")
            print(f"{'='*60}\n")
            raise e

        except Exception as e:
            print(f"[FL SongGen] ERROR loading model: {e}")
            raise e
