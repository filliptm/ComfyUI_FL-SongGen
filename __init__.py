"""
FL Song Gen - AI Song Generation for ComfyUI
Based on Tencent's SongGeneration (LeVo) model

Generate complete songs with vocals and instrumentals from lyrics!
"""

# Patch for AMD GPU users and systems without torch.distributed support
# This must run before any imports that trigger dac/audiotools
# which access torch.distributed.ReduceOp at import time
try:
    import torch.distributed as _dist
    if not hasattr(_dist, 'ReduceOp'):
        # Create a dummy ReduceOp class for non-distributed environments
        class _DummyReduceOp:
            AVG = None
            SUM = None
            PRODUCT = None
            MIN = None
            MAX = None
            BAND = None
            BOR = None
            BXOR = None
        _dist.ReduceOp = _DummyReduceOp
except ImportError:
    pass

import sys
import os
import warnings
import importlib.util

# Suppress cosmetic warnings from transformers about GenerationMixin and checkpointing format
# These need to be set early, before transformers is imported
warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*old version of the checkpointing format.*")
warnings.filterwarnings("ignore", message=".*doesn't directly inherit from.*")
warnings.filterwarnings("ignore", message=".*will NOT inherit from.*")
warnings.filterwarnings("ignore", message=".*_set_gradient_checkpointing.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def import_module_from_path(module_name, file_path):
    """Import a module from an explicit file path to avoid naming conflicts."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import utilities first (needed by nodes)
sg_paths = import_module_from_path(
    "sg_paths",
    os.path.join(current_dir, "fl_utils", "paths.py")
)
sg_audio_utils = import_module_from_path(
    "sg_audio_utils",
    os.path.join(current_dir, "fl_utils", "audio_utils.py")
)
sg_model_manager = import_module_from_path(
    "sg_model_manager",
    os.path.join(current_dir, "fl_utils", "model_manager.py")
)
sg_songgen_wrapper = import_module_from_path(
    "sg_songgen_wrapper",
    os.path.join(current_dir, "fl_utils", "songgen_wrapper.py")
)

# Import nodes
sg_model_loader = import_module_from_path(
    "sg_model_loader",
    os.path.join(current_dir, "fl_nodes", "model_loader.py")
)
sg_lyrics_formatter = import_module_from_path(
    "sg_lyrics_formatter",
    os.path.join(current_dir, "fl_nodes", "lyrics_formatter.py")
)
sg_description_builder = import_module_from_path(
    "sg_description_builder",
    os.path.join(current_dir, "fl_nodes", "description_builder.py")
)
sg_generate = import_module_from_path(
    "sg_generate",
    os.path.join(current_dir, "fl_nodes", "generate.py")
)
sg_style_transfer = import_module_from_path(
    "sg_style_transfer",
    os.path.join(current_dir, "fl_nodes", "style_transfer.py")
)
sg_auto_style = import_module_from_path(
    "sg_auto_style",
    os.path.join(current_dir, "fl_nodes", "auto_style.py")
)

# Get node classes
FL_SongGen_ModelLoader = sg_model_loader.FL_SongGen_ModelLoader
FL_SongGen_LyricsFormatter = sg_lyrics_formatter.FL_SongGen_LyricsFormatter
FL_SongGen_DescriptionBuilder = sg_description_builder.FL_SongGen_DescriptionBuilder
FL_SongGen_Generate = sg_generate.FL_SongGen_Generate
FL_SongGen_StyleTransfer = sg_style_transfer.FL_SongGen_StyleTransfer
FL_SongGen_AutoStyle = sg_auto_style.FL_SongGen_AutoStyle

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FL_SongGen_ModelLoader": FL_SongGen_ModelLoader,
    "FL_SongGen_LyricsFormatter": FL_SongGen_LyricsFormatter,
    "FL_SongGen_DescriptionBuilder": FL_SongGen_DescriptionBuilder,
    "FL_SongGen_Generate": FL_SongGen_Generate,
    "FL_SongGen_StyleTransfer": FL_SongGen_StyleTransfer,
    "FL_SongGen_AutoStyle": FL_SongGen_AutoStyle,
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_SongGen_ModelLoader": "FL Song Gen Model Loader",
    "FL_SongGen_LyricsFormatter": "FL Song Gen Lyrics Formatter",
    "FL_SongGen_DescriptionBuilder": "FL Song Gen Description Builder",
    "FL_SongGen_Generate": "FL Song Gen Generate",
    "FL_SongGen_StyleTransfer": "FL Song Gen Style Transfer",
    "FL_SongGen_AutoStyle": "FL Song Gen Auto Style",
}

# Version info
__version__ = "1.0.0"

# ASCII banner
ascii_art = """
⣏⡉ ⡇    ⢎⡑ ⢀⡀ ⣀⡀ ⢀⡀ ⡎⠑ ⢀⡀ ⣀⡀
⠇  ⠧⠤   ⠢⠜ ⠣⠜ ⠇⠸ ⣑⡺ ⠣⠝ ⠣⠭ ⠇⠸
"""
print(f"\033[35m{ascii_art}\033[0m")
print(f"FL Song Gen v{__version__} - AI Song Generation for ComfyUI")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
