"""
FL Song Gen Model Loader Node.
Loads the SongGeneration model with configurable options.
"""

import sys
import os
from typing import Tuple
import importlib.util

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
    def INPUT_TYPES(cls):
        variants = get_variant_list()
        return {
            "required": {
                "model_variant": (
                    variants,
                    {
                        "default": "songgeneration_base_new",
                        "tooltip": "Model variant to load. 'base_new' supports English+Chinese."
                    }
                ),
            },
            "optional": {
                "low_mem": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable low memory mode. Uses less VRAM but slower generation."
                    }
                ),
                "force_reload": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force reload model even if already cached."
                    }
                ),
            }
        }

    def load_model(
        self,
        model_variant: str,
        low_mem: bool = False,
        force_reload: bool = False
    ) -> Tuple[dict]:
        """
        Load the SongGeneration model.

        Args:
            model_variant: Which model variant to load
            low_mem: Enable low memory mode
            force_reload: Force reload even if cached

        Returns:
            Tuple containing the model info dict
        """
        # Get variant info for logging
        variant_info = get_variant_info(model_variant)
        print(f"\n{'='*60}")
        print(f"[FL SongGen] Loading Model")
        print(f"{'='*60}")
        print(f"Variant: {model_variant}")
        print(f"Description: {variant_info['description']}")
        print(f"Max Duration: {variant_info['max_duration']}s ({variant_info['max_duration']//60}m {variant_info['max_duration']%60}s)")
        print(f"Languages: {', '.join(variant_info['languages'])}")
        print(f"VRAM Required: {variant_info['vram_low'] if low_mem else variant_info['vram_normal']}GB")
        print(f"Low Memory Mode: {low_mem}")
        print(f"{'='*60}\n")

        try:
            model_info = load_model(
                variant=model_variant,
                low_mem=low_mem,
                use_flash_attn=False,
                force_reload=force_reload
            )
            print(f"[FL SongGen] Model loaded successfully!")
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
