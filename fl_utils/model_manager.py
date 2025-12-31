"""
Model management for FL Song Gen.
Handles model loading, caching, and configuration.
Uses bundled code (codeclm, third_party) - only downloads model checkpoints.
"""

import os
import sys
import gc
import importlib.util
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Suppress cosmetic warnings from transformers about GenerationMixin and checkpointing format
warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*old version of the checkpointing format.*")

# Get the fl_utils directory (same directory as this file)
_FL_UTILS_DIR = os.path.dirname(__file__)

# Import paths module explicitly from our package to avoid conflicts
def _import_paths():
    """Import paths module from our fl_utils directory specifically."""
    module_path = os.path.join(_FL_UTILS_DIR, "paths.py")
    spec = importlib.util.spec_from_file_location("songgen_paths", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_paths = _import_paths()
get_model_variant_dir = _paths.get_model_variant_dir
get_package_root = _paths.get_package_root
get_auto_prompts_path = _paths.get_auto_prompts_path
get_demucs_dir = _paths.get_demucs_dir
get_checkpoints_dir = _paths.get_checkpoints_dir
get_songgen_models_dir = _paths.get_songgen_models_dir
check_model_files = _paths.check_model_files
check_bundled_files = _paths.check_bundled_files
check_checkpoint_files = _paths.check_checkpoint_files
setup_bundled_imports = _paths.setup_bundled_imports
get_bundled_third_party_path = _paths.get_bundled_third_party_path

# Model variant configurations with HuggingFace repo info
MODEL_VARIANTS = {
    "songgeneration_base": {
        "max_duration": 150,  # 2m30s in seconds
        "vram_normal": 16,  # GB
        "vram_low": 10,
        "languages": ["zh"],
        "description": "Base model - Chinese only, 2m30s max",
        "hf_repo": "tencent/SongGeneration",
        "hf_subfolder": "ckpt/songgeneration_base",
    },
    "songgeneration_base_new": {
        "max_duration": 150,
        "vram_normal": 16,
        "vram_low": 10,
        "languages": ["zh", "en"],
        "description": "Base model - Chinese + English, 2m30s max",
        "hf_repo": "lglg666/SongGeneration-base-new",
        "hf_subfolder": None,
    },
    "songgeneration_base_full": {
        "max_duration": 270,  # 4m30s
        "vram_normal": 18,
        "vram_low": 12,
        "languages": ["zh", "en"],
        "description": "Full base model - Chinese + English, 4m30s max",
        "hf_repo": "lglg666/SongGeneration-base-full",
        "hf_subfolder": None,
    },
    "songgeneration_large": {
        "max_duration": 270,
        "vram_normal": 28,
        "vram_low": 22,
        "languages": ["zh", "en"],
        "description": "Large model - Best quality, 4m30s max",
        "hf_repo": "lglg666/SongGeneration-large",
        "hf_subfolder": None,
    }
}

# Checkpoints HuggingFace repo - contains only model weights (no code)
CHECKPOINTS_HF_REPO = "lglg666/SongGeneration-Runtime"

AUTO_STYLE_PRESETS = [
    "Pop", "R&B", "Dance", "Jazz", "Folk",
    "Rock", "Chinese Style", "Chinese Tradition",
    "Metal", "Reggae", "Chinese Opera", "Auto"
]

# Global model cache
_MODEL_CACHE: Dict[str, Any] = {}


def get_variant_list() -> list:
    """Get list of available model variants."""
    return list(MODEL_VARIANTS.keys())


def get_variant_info(variant: str) -> dict:
    """Get information about a model variant."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(MODEL_VARIANTS.keys())}")
    return MODEL_VARIANTS[variant].copy()


def clear_model_cache():
    """Clear all cached models and free memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _setup_songgen_imports():
    """Set up Python path for bundled code imports."""
    setup_bundled_imports()


def _register_omegaconf_resolvers():
    """Register OmegaConf resolvers needed for config loading."""
    package_root = get_package_root()

    def load_yaml_resolver(path: str):
        """Load YAML file, resolving paths relative to package root."""
        # If path is relative (like conf/vocab.yaml), resolve from package root
        if not os.path.isabs(path):
            resolved_path = os.path.join(package_root, path)
        else:
            resolved_path = path
        return list(OmegaConf.load(resolved_path))

    try:
        OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
        OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx], replace=True)
        OmegaConf.register_new_resolver("get_fname", lambda: "songgen", replace=True)
        OmegaConf.register_new_resolver("load_yaml", load_yaml_resolver, replace=True)
    except Exception:
        # Resolvers may already be registered
        pass


def _download_model_files(variant: str) -> bool:
    """
    Download model files from HuggingFace.

    Args:
        variant: Model variant name

    Returns:
        True if download successful
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[FL SongGen] ERROR: huggingface_hub not installed. Please run: pip install huggingface-hub")
        return False

    if variant not in MODEL_VARIANTS:
        print(f"[FL SongGen] ERROR: Unknown variant: {variant}")
        return False

    variant_info = MODEL_VARIANTS[variant]
    hf_repo = variant_info["hf_repo"]
    hf_subfolder = variant_info.get("hf_subfolder")
    target_dir = get_model_variant_dir(variant)

    print(f"[FL SongGen] Downloading model files for {variant}...")
    print(f"[FL SongGen] From: {hf_repo}")
    print(f"[FL SongGen] To: {target_dir}")

    try:
        # Download only the specific files we need: config.yaml and model.pt
        for filename in ["config.yaml", "model.pt"]:
            target_file = target_dir / filename
            if target_file.exists():
                print(f"[FL SongGen] {filename} already exists, skipping...")
                continue

            if hf_subfolder:
                # File is in a subfolder of the repo
                filepath = f"{hf_subfolder}/{filename}"
            else:
                # File is at root of repo
                filepath = filename

            print(f"[FL SongGen] Downloading {filepath}...")
            hf_hub_download(
                repo_id=hf_repo,
                filename=filepath,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )

            # If downloaded to subfolder, move to root
            if hf_subfolder:
                import shutil
                src = target_dir / filepath
                if src.exists() and src != target_file:
                    shutil.move(str(src), str(target_file))
                    # Clean up empty subfolder
                    subfolder_path = target_dir / hf_subfolder.split('/')[0]
                    if subfolder_path.exists() and subfolder_path.is_dir():
                        shutil.rmtree(str(subfolder_path))

        print(f"[FL SongGen] Model download complete!")
        return True

    except Exception as e:
        print(f"[FL SongGen] ERROR downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def _download_checkpoint_files() -> bool:
    """
    Download checkpoint files (model weights only) from HuggingFace.
    Code is bundled in the node pack, so we only need the weights.

    Returns:
        True if download successful
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[FL SongGen] ERROR: huggingface_hub not installed. Please run: pip install huggingface-hub")
        return False

    ckpt_dir = get_checkpoints_dir()

    # Check if checkpoints already exist
    ckpt_check = check_checkpoint_files()
    if ckpt_check['exists']:
        print("[FL SongGen] Checkpoint files already exist")
        return True

    print(f"[FL SongGen] Downloading checkpoint files...")
    print(f"[FL SongGen] From: {CHECKPOINTS_HF_REPO}")
    print(f"[FL SongGen] To: {ckpt_dir}")
    print("[FL SongGen] This may take a while (several GB of data)...")

    try:
        # Download only checkpoint files (model weights), not code
        snapshot_download(
            repo_id=CHECKPOINTS_HF_REPO,
            local_dir=str(ckpt_dir.parent),  # Download to songgen dir
            local_dir_use_symlinks=False,
            allow_patterns=["ckpt/**"],  # Only get checkpoint files
            ignore_patterns=["*.md", ".gitattributes", ".git*", "third_party/**", "codeclm/**"],
        )

        print(f"[FL SongGen] Checkpoint download complete!")
        return True

    except Exception as e:
        print(f"[FL SongGen] ERROR downloading checkpoints: {e}")
        import traceback
        traceback.print_exc()
        return False


def ensure_model_files(variant: str) -> bool:
    """
    Ensure model files exist, downloading if necessary.

    Args:
        variant: Model variant name

    Returns:
        True if files exist or were downloaded successfully
    """
    # First verify bundled code exists
    bundled_check = check_bundled_files()
    if not bundled_check['exists']:
        print(f"[FL SongGen] ERROR: Bundled code files missing: {bundled_check['missing']}")
        print("[FL SongGen] The node pack installation may be corrupted. Please reinstall.")
        return False

    # Check if checkpoint files exist, download if needed
    ckpt_check = check_checkpoint_files()
    if not ckpt_check['exists']:
        print(f"[FL SongGen] Checkpoint files missing: {ckpt_check['missing']}")
        print("[FL SongGen] Downloading checkpoint files...")
        if not _download_checkpoint_files():
            return False

    # Check model variant files
    file_check = check_model_files(variant)
    if file_check['exists']:
        return True

    # Download model variant files
    print(f"[FL SongGen] Model files missing: {file_check['missing']}")
    return _download_model_files(variant)


def load_model(
    variant: str,
    low_mem: bool = False,
    use_flash_attn: bool = False,
    force_reload: bool = False,
    device: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Load SongGeneration model.

    Args:
        variant: Model variant name
        low_mem: Enable low memory mode
        use_flash_attn: Use Flash Attention 2
        force_reload: Force reload even if cached
        device: Device to load model on (default: auto-detect)
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        Dict containing model components and configuration
    """
    global _MODEL_CACHE

    cache_key = f"{variant}_{low_mem}_{use_flash_attn}"

    # Return cached model if available
    if not force_reload and cache_key in _MODEL_CACHE:
        print(f"[FL SongGen] Using cached model: {variant}")
        return _MODEL_CACHE[cache_key]

    # Clear cache if force reload
    if force_reload:
        clear_model_cache()

    # Ensure model files exist (download if necessary)
    if not ensure_model_files(variant):
        file_check = check_model_files(variant)
        raise FileNotFoundError(
            f"Model files missing for {variant} at {file_check['path']}. "
            f"Missing: {file_check['missing']}. "
            "Automatic download failed. Please download manually from HuggingFace."
        )

    print(f"[FL SongGen] Loading model: {variant}")
    print(f"[FL SongGen] Low memory mode: {low_mem}")
    print(f"[FL SongGen] Flash Attention: {use_flash_attn}")

    # Setup imports from bundled code
    _setup_songgen_imports()
    _register_omegaconf_resolvers()

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            print("[FL SongGen] WARNING: CUDA not available, using CPU (very slow)")

    # Load configuration
    variant_dir = get_model_variant_dir(variant)
    cfg_path = variant_dir / "config.yaml"
    ckpt_path = variant_dir / "model.pt"

    cfg = OmegaConf.load(str(cfg_path))
    cfg.lm.use_flash_attn_2 = use_flash_attn
    cfg.mode = 'inference'
    max_duration = cfg.max_dur

    # Resolve relative paths in config to absolute paths
    # The config uses paths like ./ckpt/... which need to be relative to songgen models dir
    models_dir = get_songgen_models_dir()
    bundled_third_party = get_bundled_third_party_path()

    def resolve_path(path_str: str) -> str:
        """Resolve relative paths in config to absolute paths."""
        if path_str.startswith('./ckpt/') or path_str.startswith('ckpt/'):
            # Checkpoint paths: resolve relative to models/songgen/
            rel_path = path_str.lstrip('./')
            return str(models_dir / rel_path)
        elif path_str.startswith('third_party/'):
            # Third party paths: resolve to bundled third_party
            rel_path = path_str.replace('third_party/', '')
            return str(bundled_third_party / rel_path)
        elif path_str.startswith('./'):
            # Other relative paths: resolve to models dir
            return str(models_dir / path_str.lstrip('./'))
        return path_str

    # Update config paths
    if hasattr(cfg, 'vae_config'):
        cfg.vae_config = resolve_path(cfg.vae_config)
    if hasattr(cfg, 'vae_model'):
        cfg.vae_model = resolve_path(cfg.vae_model)
    if hasattr(cfg, 'audio_tokenizer_checkpoint'):
        # Format: Type_path - only resolve the path part
        parts = cfg.audio_tokenizer_checkpoint.split('_', 1)
        if len(parts) == 2:
            cfg.audio_tokenizer_checkpoint = f"{parts[0]}_{resolve_path(parts[1])}"
    if hasattr(cfg, 'audio_tokenizer_checkpoint_sep'):
        parts = cfg.audio_tokenizer_checkpoint_sep.split('_', 1)
        if len(parts) == 2:
            cfg.audio_tokenizer_checkpoint_sep = f"{parts[0]}_{resolve_path(parts[1])}"

    # Update conditioner paths
    if hasattr(cfg, 'conditioners'):
        for cond_name, cond_cfg in cfg.conditioners.items():
            if hasattr(cond_cfg, 'QwTokenizer') and hasattr(cond_cfg.QwTokenizer, 'token_path'):
                cond_cfg.QwTokenizer.token_path = resolve_path(cond_cfg.QwTokenizer.token_path)
            if hasattr(cond_cfg, 'QwTextTokenizer') and hasattr(cond_cfg.QwTextTokenizer, 'token_path'):
                cond_cfg.QwTextTokenizer.token_path = resolve_path(cond_cfg.QwTextTokenizer.token_path)

    # Import from bundled code
    from codeclm.models import builders, CodecLM

    model_info = {
        "variant": variant,
        "config": cfg,
        "max_duration": max_duration,
        "sample_rate": cfg.sample_rate,
        "device": device,
        "low_mem": low_mem,
        "use_flash_attn": use_flash_attn,
    }

    if low_mem:
        # Low memory mode: load components on-demand
        model_info["ckpt_path"] = str(ckpt_path)
        model_info["loaded"] = False
        print("[FL SongGen] Low memory mode: model will be loaded on-demand")
        if progress_callback:
            progress_callback(1, 1)  # Complete immediately for low_mem mode
    else:
        # Normal mode: load everything now
        model_info = _load_full_model(model_info, cfg, ckpt_path, device, progress_callback)

    # Load auto prompts if available
    auto_prompts_path = get_auto_prompts_path()
    if auto_prompts_path.exists():
        model_info["auto_prompts"] = torch.load(str(auto_prompts_path), map_location='cpu')
        print(f"[FL SongGen] Loaded auto prompts from {auto_prompts_path}")
    else:
        model_info["auto_prompts"] = None
        print(f"[FL SongGen] Auto prompts not found at {auto_prompts_path}")

    # Cache the model
    _MODEL_CACHE[cache_key] = model_info

    print(f"[FL SongGen] Model loaded successfully")
    return model_info


def _load_full_model(
    model_info: dict,
    cfg: OmegaConf,
    ckpt_path: Path,
    device: str,
    progress_callback: Optional[callable] = None
) -> dict:
    """Load full model (non-low-memory mode)."""
    from codeclm.models import builders, CodecLM

    # Total steps: audio_tokenizer, separate_tokenizer, language_model, create_wrapper
    total_steps = 4
    current_step = 0

    def update_progress():
        nonlocal current_step
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps)

    # Load audio tokenizer for prompt encoding
    print("[FL SongGen] Loading audio tokenizer...")
    audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
    if audio_tokenizer is not None:
        audio_tokenizer = audio_tokenizer.eval()
        if device == "cuda":
            audio_tokenizer = audio_tokenizer.cuda()
    model_info["audio_tokenizer"] = audio_tokenizer
    update_progress()

    # Load separate tokenizer for vocal/bgm encoding
    print("[FL SongGen] Loading separate tokenizer...")
    if "audio_tokenizer_checkpoint_sep" in cfg.keys():
        separate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
        if separate_tokenizer is not None:
            separate_tokenizer = separate_tokenizer.eval()
            if device == "cuda":
                separate_tokenizer = separate_tokenizer.cuda()
    else:
        separate_tokenizer = None
    model_info["separate_tokenizer"] = separate_tokenizer
    update_progress()

    # Load LM
    print("[FL SongGen] Loading language model...")
    audiolm = builders.get_lm_model(cfg)
    checkpoint = torch.load(str(ckpt_path), map_location='cpu')
    audiolm_state_dict = {
        k.replace('audiolm.', ''): v
        for k, v in checkpoint.items()
        if k.startswith('audiolm')
    }

    # Resize embedding layers that have size mismatches (due to tokenizer version differences)
    def get_nested_attr(obj, attr_path):
        """Get nested attribute from object using dot-separated path."""
        parts = attr_path.split('.')
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif hasattr(obj, '_modules') and part in obj._modules:
                obj = obj._modules[part]
            else:
                return None
        return obj

    def set_nested_attr(obj, attr_path, value):
        """Set nested attribute on object using dot-separated path."""
        parts = attr_path.split('.')
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif hasattr(obj, '_modules') and part in obj._modules:
                obj = obj._modules[part]
            else:
                return False
        setattr(obj, parts[-1], value)
        return True

    for key, ckpt_tensor in audiolm_state_dict.items():
        if 'output_proj.weight' in key:
            # Get the path to the parent module (remove .weight)
            module_path = key.rsplit('.', 1)[0]
            current_module = get_nested_attr(audiolm, module_path)

            if current_module is not None and hasattr(current_module, 'weight'):
                current_size = current_module.weight.shape[0]
                checkpoint_size = ckpt_tensor.shape[0]

                if current_size != checkpoint_size:
                    print(f"[FL SongGen] Resizing embedding {module_path}: {current_size} -> {checkpoint_size}")
                    # Create new embedding with checkpoint size
                    embed_dim = ckpt_tensor.shape[1]
                    padding_idx = getattr(current_module, 'padding_idx', None)
                    new_embedding = nn.Embedding(checkpoint_size, embed_dim, padding_idx=padding_idx)
                    set_nested_attr(audiolm, module_path, new_embedding)

    audiolm.load_state_dict(audiolm_state_dict, strict=False)
    audiolm = audiolm.eval()

    if device == "cuda":
        audiolm = audiolm.cuda().to(torch.float16)
    update_progress()

    # Create CodecLM wrapper
    print("[FL SongGen] Creating model wrapper...")
    model = CodecLM(
        name=model_info["variant"],
        lm=audiolm,
        audiotokenizer=audio_tokenizer,
        max_duration=model_info["max_duration"],
        seperate_tokenizer=separate_tokenizer,
    )

    model_info["model"] = model
    model_info["audiolm"] = audiolm
    model_info["loaded"] = True
    update_progress()

    # Cleanup checkpoint to save memory
    del checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model_info


def _download_demucs_model() -> bool:
    """
    Download Demucs htdemucs model from HuggingFace.

    Returns:
        True if download successful
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[FL SongGen] ERROR: huggingface_hub not installed.")
        return False

    demucs_dir = get_demucs_dir()
    demucs_dir.mkdir(parents=True, exist_ok=True)
    dm_model_path = demucs_dir / "htdemucs.pth"

    if dm_model_path.exists():
        return True

    print("[FL SongGen] Downloading Demucs model for audio separation...")
    print("[FL SongGen] This is required for style transfer functionality.")

    try:
        # Download from the official tencent/SongGeneration repo
        # The file is at third_party/demucs/ckpt/htdemucs.pth
        downloaded_path = hf_hub_download(
            repo_id="tencent/SongGeneration",
            filename="third_party/demucs/ckpt/htdemucs.pth",
            local_dir_use_symlinks=False,
        )
        # Move/copy to the expected location
        import shutil
        shutil.copy2(downloaded_path, str(dm_model_path))
        print("[FL SongGen] Demucs model downloaded successfully!")
        return True
    except Exception as e:
        print(f"[FL SongGen] ERROR downloading Demucs model: {e}")
        print("[FL SongGen] Please manually download htdemucs.pth from:")
        print("[FL SongGen] https://huggingface.co/tencent/SongGeneration/blob/main/third_party/demucs/ckpt/htdemucs.pth")
        print(f"[FL SongGen] And place it at: {dm_model_path}")
        return False


def load_separator(device: str = "cuda") -> Any:
    """
    Load Demucs separator for audio source separation.

    Args:
        device: Device to load on

    Returns:
        Separator instance
    """
    _setup_songgen_imports()

    # Import from bundled third_party
    from third_party.demucs.models.pretrained import get_model_from_yaml

    demucs_dir = get_demucs_dir()
    dm_model_path = demucs_dir / "htdemucs.pth"

    # Config is bundled in third_party
    bundled_demucs = get_bundled_third_party_path() / "demucs" / "ckpt"
    dm_config_path = bundled_demucs / "htdemucs.yaml"

    # Auto-download if missing
    if not dm_model_path.exists():
        if not _download_demucs_model():
            raise FileNotFoundError(
                f"Demucs model not found at {dm_model_path}. "
                "Automatic download failed. Please download manually."
            )

    demucs_model = get_model_from_yaml(str(dm_config_path), str(dm_model_path))

    if device == "cuda" and torch.cuda.is_available():
        demucs_model = demucs_model.to(torch.device("cuda"))
    else:
        demucs_model = demucs_model.to(torch.device("cpu"))

    demucs_model.eval()

    print("[FL SongGen] Demucs separator loaded")
    return demucs_model


def get_model_status() -> dict:
    """Get status of loaded models."""
    status = {
        "cached_models": list(_MODEL_CACHE.keys()),
        "available_variants": list(MODEL_VARIANTS.keys()),
        "auto_style_presets": AUTO_STYLE_PRESETS,
        "bundled_code_status": check_bundled_files(),
        "checkpoint_status": check_checkpoint_files(),
    }

    # Check VRAM if CUDA available
    if torch.cuda.is_available():
        status["cuda_available"] = True
        status["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        status["vram_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
        status["vram_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
    else:
        status["cuda_available"] = False

    return status
