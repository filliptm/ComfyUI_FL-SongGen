"""
SongGeneration wrapper for FL Song Gen.
Encapsulates the inference logic from SongGeneration's generate.py.
"""

import gc
import os
import re
import sys
import tempfile
import importlib.util
from typing import Optional, Tuple, Callable, Any, Dict

import numpy as np
import torch
import torchaudio
import soundfile as sf

# Regex to filter lyrics - keeps letters, numbers, whitespace, brackets, hyphens, and CJK characters
# Removes punctuation like commas, apostrophes, quotes, etc. that the model wasn't trained on
LYRICS_FILTER_REGEX = re.compile(
    r"[^\w\s\[\]\-;\.\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u00c0-\u017f]"
)

# Get the fl_utils directory (same directory as this file)
_FL_UTILS_DIR = os.path.dirname(__file__)

# Import modules explicitly from our package to avoid conflicts with other FL packages
def _import_from_fl_utils(module_name, file_name):
    """Import a module from our fl_utils directory specifically."""
    module_path = os.path.join(_FL_UTILS_DIR, f"{file_name}.py")
    spec = importlib.util.spec_from_file_location(f"songgen_{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import our modules
_paths = _import_from_fl_utils("paths", "paths")
_audio_utils = _import_from_fl_utils("audio_utils", "audio_utils")

get_songgen_repo_path = _paths.get_songgen_repo_path
tensor_to_comfyui_audio = _audio_utils.tensor_to_comfyui_audio
empty_audio = _audio_utils.empty_audio


class SongGenWrapper:
    """
    Wrapper for SongGeneration model inference.
    Adapts generate.py logic for ComfyUI integration.
    """

    def __init__(self, model_info: dict):
        """
        Initialize wrapper with loaded model info.

        Args:
            model_info: Dict from model_manager.load_model()
        """
        self.model_info = model_info
        self.config = model_info["config"]
        self.max_duration = model_info["max_duration"]
        self.sample_rate = model_info.get("sample_rate", 24000)
        self.device = model_info.get("device", "cuda")
        self.low_mem = model_info.get("low_mem", False)
        self.ultra_low_mem = model_info.get("ultra_low_mem", False)
        self.auto_prompts = model_info.get("auto_prompts")

        # Frame rate for progress calculation
        self.frame_rate = 25

        # Progress callback
        self._progress_callback: Optional[Callable[[int, int], None]] = None

    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback

    def generate(
        self,
        lyrics: str,
        description: Optional[str] = None,
        prompt_audio: Optional[torch.Tensor] = None,
        auto_style: Optional[str] = None,
        duration: float = 150.0,
        temperature: float = 0.9,
        cfg_coef: float = 1.5,
        top_k: int = 50,
        gen_type: str = "mixed",
        seed: int = -1,
    ) -> Tuple[dict, Optional[dict], Optional[dict]]:
        """
        Generate song from lyrics with conditioning.

        Args:
            lyrics: Formatted lyrics string with section tags
            description: Style description (e.g., "female, pop, sad, piano")
            prompt_audio: Reference audio tensor [B, C, S] at 48kHz for style transfer
            auto_style: Preset style name from AUTO_STYLE_PRESETS
            duration: Target duration in seconds
            temperature: Sampling temperature
            cfg_coef: Classifier-free guidance coefficient
            top_k: Top-k sampling parameter
            gen_type: "mixed", "vocal", "bgm", or "separate"
            seed: Random seed (-1 for random)

        Returns:
            (mixed_audio, vocal_audio, bgm_audio) as ComfyUI AUDIO dicts
            vocal_audio and bgm_audio only present if gen_type="separate"
        """
        # Handle seed
        if seed == -1:
            seed = int(np.random.randint(0, 2147483647))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Validate duration
        if duration > self.max_duration:
            print(f"[FL SongGen] Duration {duration}s exceeds max {self.max_duration}s, clamping.")
            duration = self.max_duration

        # Clean lyrics - remove unsupported punctuation and normalize spaces
        lyrics = LYRICS_FILTER_REGEX.sub("", lyrics)
        lyrics = re.sub(r"\s+", " ", lyrics)  # Normalize multiple spaces to single space

        if self.ultra_low_mem:
            return self._generate_ultra_lowmem(
                lyrics, description, prompt_audio, auto_style,
                duration, temperature, cfg_coef, top_k, gen_type
            )
        elif self.low_mem:
            return self._generate_lowmem(
                lyrics, description, prompt_audio, auto_style,
                duration, temperature, cfg_coef, top_k, gen_type
            )
        else:
            return self._generate_normal(
                lyrics, description, prompt_audio, auto_style,
                duration, temperature, cfg_coef, top_k, gen_type
            )

    def _generate_normal(
        self,
        lyrics: str,
        description: Optional[str],
        prompt_audio: Optional[torch.Tensor],
        auto_style: Optional[str],
        duration: float,
        temperature: float,
        cfg_coef: float,
        top_k: int,
        gen_type: str,
    ) -> Tuple[dict, Optional[dict], Optional[dict]]:
        """Normal generation mode (sufficient VRAM)."""
        model = self.model_info["model"]
        audio_tokenizer = self.model_info.get("audio_tokenizer")
        separate_tokenizer = self.model_info.get("separate_tokenizer")

        # Prepare prompt tokens
        pmt_wav, vocal_wav, bgm_wav, melody_is_wav = self._prepare_prompts(
            prompt_audio, auto_style, audio_tokenizer
        )

        # Store raw wavs for audio generation if using prompt audio
        raw_pmt_wav = None
        raw_vocal_wav = None
        raw_bgm_wav = None

        if prompt_audio is not None and separate_tokenizer is not None:
            raw_pmt_wav, raw_vocal_wav, raw_bgm_wav = self._separate_audio(prompt_audio)
            # Encode vocal and bgm with separate tokenizer
            with torch.no_grad():
                vocal_wav, bgm_wav = separate_tokenizer.encode(
                    raw_vocal_wav.to(self.device),
                    raw_bgm_wav.to(self.device)
                )

        # Set generation parameters
        model.set_generation_params(
            duration=duration,
            extend_stride=5,
            temperature=temperature,
            cfg_coef=cfg_coef,
            top_k=top_k,
            top_p=0.0,
            record_tokens=True,
            record_window=50
        )

        # Set progress callback - pass through actual total from model
        def progress_wrapper(current, total):
            if self._progress_callback:
                self._progress_callback(current, total)

        model.set_custom_progress_callback(progress_wrapper)

        # Generate tokens
        generate_inp = {
            'lyrics': [lyrics],
            'descriptions': [description],
            'melody_wavs': pmt_wav,
            'vocal_wavs': vocal_wav,
            'bgm_wavs': bgm_wav,
            'melody_is_wav': melody_is_wav,
        }

        # Debug: Log what we're passing to the model (normal mode)
        print(f"\n[FL SongGen DEBUG] ========== GENERATION INPUT ==========")
        print(f"[FL SongGen DEBUG] Lyrics (first 200 chars): {repr(lyrics[:200]) if lyrics else 'None'}")
        print(f"[FL SongGen DEBUG] Description: {repr(description) if description else 'None'}")
        print(f"[FL SongGen DEBUG] melody_wavs shape: {pmt_wav.shape if pmt_wav is not None else 'None'}")
        print(f"[FL SongGen DEBUG] vocal_wavs shape: {vocal_wav.shape if vocal_wav is not None else 'None'}")
        print(f"[FL SongGen DEBUG] bgm_wavs shape: {bgm_wav.shape if bgm_wav is not None else 'None'}")
        print(f"[FL SongGen DEBUG] melody_is_wav: {melody_is_wav}")
        print(f"[FL SongGen DEBUG] ======================================\n")

        print(f"[FL SongGen] Generating tokens for {duration}s song...")

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            with torch.no_grad():
                tokens = model.generate(**generate_inp, return_tokens=True)

        # Generate audio from tokens
        print(f"[FL SongGen] Decoding audio...")

        with torch.no_grad():
            if gen_type == 'separate':
                # Generate all three tracks
                if raw_pmt_wav is not None:
                    wav_mixed = model.generate_audio(
                        tokens, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav,
                        chunked=True, gen_type='mixed'
                    )
                    wav_vocal = model.generate_audio(
                        tokens, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav,
                        chunked=True, gen_type='vocal'
                    )
                    wav_bgm = model.generate_audio(
                        tokens, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav,
                        chunked=True, gen_type='bgm'
                    )
                else:
                    wav_mixed = model.generate_audio(tokens, chunked=True, gen_type='mixed')
                    wav_vocal = model.generate_audio(tokens, chunked=True, gen_type='vocal')
                    wav_bgm = model.generate_audio(tokens, chunked=True, gen_type='bgm')
            else:
                # Generate single track
                if raw_pmt_wav is not None:
                    wav_mixed = model.generate_audio(
                        tokens, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav,
                        chunked=True, gen_type=gen_type
                    )
                else:
                    wav_mixed = model.generate_audio(tokens, chunked=True, gen_type=gen_type)
                wav_vocal = None
                wav_bgm = None

        # Convert to ComfyUI format
        mixed_audio = tensor_to_comfyui_audio(wav_mixed[0].cpu().float(), self.sample_rate)

        if gen_type == 'separate':
            vocal_audio = tensor_to_comfyui_audio(wav_vocal[0].cpu().float(), self.sample_rate)
            bgm_audio = tensor_to_comfyui_audio(wav_bgm[0].cpu().float(), self.sample_rate)
        else:
            vocal_audio = empty_audio(self.sample_rate)
            bgm_audio = empty_audio(self.sample_rate)

        print(f"[FL SongGen] Generation complete!")
        return mixed_audio, vocal_audio, bgm_audio

    def _generate_lowmem(
        self,
        lyrics: str,
        description: Optional[str],
        prompt_audio: Optional[torch.Tensor],
        auto_style: Optional[str],
        duration: float,
        temperature: float,
        cfg_coef: float,
        top_k: int,
        gen_type: str,
    ) -> Tuple[dict, Optional[dict], Optional[dict]]:
        """Low memory generation mode (limited VRAM)."""
        # Import builders for on-demand loading
        from codeclm.models import builders, CodecLM

        cfg = self.config
        ckpt_path = self.model_info["ckpt_path"]

        # Determine if we need audio tokenizer
        use_audio_tokenizer = prompt_audio is not None

        # Store raw wavs for later
        raw_pmt_wav = None
        raw_vocal_wav = None
        raw_bgm_wav = None

        # Phase 1: Process prompts with audio tokenizer
        if use_audio_tokenizer:
            print("[FL SongGen LowMem] Loading audio tokenizer...")
            audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
            audio_tokenizer = audio_tokenizer.eval().cuda()

            # Separate and encode prompt audio
            raw_pmt_wav, raw_vocal_wav, raw_bgm_wav = self._separate_audio(prompt_audio)

            with torch.no_grad():
                pmt_wav, _ = audio_tokenizer.encode(raw_pmt_wav.cuda())

            del audio_tokenizer
            torch.cuda.empty_cache()

            # Load separate tokenizer for vocal/bgm
            print("[FL SongGen LowMem] Loading separate tokenizer...")
            separate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
            separate_tokenizer = separate_tokenizer.eval().cuda()

            with torch.no_grad():
                vocal_wav, bgm_wav = separate_tokenizer.encode(
                    raw_vocal_wav.cuda(),
                    raw_bgm_wav.cuda()
                )

            del separate_tokenizer
            torch.cuda.empty_cache()

            melody_is_wav = False
        else:
            # Use auto prompts or no prompt
            pmt_wav, vocal_wav, bgm_wav, melody_is_wav = self._prepare_prompts(
                None, auto_style, None
            )

        # Phase 2: Generate tokens with LM
        print("[FL SongGen LowMem] Loading language model...")
        audiolm = builders.get_lm_model(cfg)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        audiolm_state_dict = {
            k.replace('audiolm.', ''): v
            for k, v in checkpoint.items()
            if k.startswith('audiolm')
        }

        # Resize embedding layers to match checkpoint (fixes tokenizer version mismatch)
        audiolm = self._resize_embeddings_for_checkpoint(audiolm, audiolm_state_dict)

        audiolm.load_state_dict(audiolm_state_dict, strict=False)
        audiolm = audiolm.eval().cuda().to(torch.float16)

        model = CodecLM(
            name="tmp",
            lm=audiolm,
            audiotokenizer=None,
            max_duration=self.max_duration,
            seperate_tokenizer=None,
        )

        model.set_generation_params(
            duration=duration,
            extend_stride=5,
            temperature=temperature,
            cfg_coef=cfg_coef,
            top_k=top_k,
            top_p=0.0,
            record_tokens=True,
            record_window=50
        )

        # Set progress callback - pass through actual total from model
        def progress_wrapper(current, total):
            if self._progress_callback:
                self._progress_callback(current, total)

        model.set_custom_progress_callback(progress_wrapper)

        generate_inp = {
            'lyrics': [lyrics],
            'descriptions': [description],
            'melody_wavs': pmt_wav,
            'vocal_wavs': vocal_wav,
            'bgm_wavs': bgm_wav,
            'melody_is_wav': melody_is_wav,
        }

        # Debug: Log what we're passing to the model (low mem path)
        print(f"\n[FL SongGen LowMem DEBUG] ========== GENERATION INPUT ==========")
        print(f"[FL SongGen LowMem DEBUG] Lyrics (first 200 chars): {repr(lyrics[:200]) if lyrics else 'None'}")
        print(f"[FL SongGen LowMem DEBUG] Description: {repr(description) if description else 'None'}")
        print(f"[FL SongGen LowMem DEBUG] melody_wavs shape: {pmt_wav.shape if pmt_wav is not None else 'None'}")
        print(f"[FL SongGen LowMem DEBUG] vocal_wavs shape: {vocal_wav.shape if vocal_wav is not None else 'None'}")
        print(f"[FL SongGen LowMem DEBUG] bgm_wavs shape: {bgm_wav.shape if bgm_wav is not None else 'None'}")
        print(f"[FL SongGen LowMem DEBUG] melody_is_wav: {melody_is_wav}")
        print(f"[FL SongGen LowMem DEBUG] ======================================\n")

        print(f"[FL SongGen LowMem] Generating tokens...")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                tokens = model.generate(**generate_inp, return_tokens=True)

        # Cleanup LM
        del model
        audiolm = audiolm.cpu()
        del audiolm
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

        # Phase 3: Decode audio with separate tokenizer
        print("[FL SongGen LowMem] Loading audio decoder...")
        separate_tokenizer = builders.get_audio_tokenizer_model_cpu(cfg.audio_tokenizer_checkpoint_sep, cfg)
        device = "cuda:0"
        separate_tokenizer.model.device = device
        separate_tokenizer.model.vae = separate_tokenizer.model.vae.to(device)
        separate_tokenizer.model.model.device = torch.device(device)
        separate_tokenizer.model.model = separate_tokenizer.model.model.to(device)
        separate_tokenizer = separate_tokenizer.eval()

        model = CodecLM(
            name="tmp",
            lm=None,
            audiotokenizer=None,
            max_duration=self.max_duration,
            seperate_tokenizer=separate_tokenizer,
        )

        print(f"[FL SongGen LowMem] Decoding audio...")
        with torch.no_grad():
            if gen_type == 'separate':
                if raw_pmt_wav is not None:
                    wav_mixed = model.generate_audio(
                        tokens, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav,
                        chunked=True, gen_type='mixed'
                    )
                    wav_vocal = model.generate_audio(
                        tokens, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav,
                        chunked=True, gen_type='vocal'
                    )
                    wav_bgm = model.generate_audio(
                        tokens, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav,
                        chunked=True, gen_type='bgm'
                    )
                else:
                    wav_mixed = model.generate_audio(tokens, chunked=True, gen_type='mixed')
                    wav_vocal = model.generate_audio(tokens, chunked=True, gen_type='vocal')
                    wav_bgm = model.generate_audio(tokens, chunked=True, gen_type='bgm')
            else:
                if raw_pmt_wav is not None:
                    wav_mixed = model.generate_audio(
                        tokens, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav,
                        chunked=True, gen_type=gen_type
                    )
                else:
                    wav_mixed = model.generate_audio(tokens, chunked=True, gen_type=gen_type)
                wav_vocal = None
                wav_bgm = None

        # Cleanup
        del model
        del separate_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # Convert to ComfyUI format
        mixed_audio = tensor_to_comfyui_audio(wav_mixed[0].cpu().float(), self.sample_rate)

        if gen_type == 'separate':
            vocal_audio = tensor_to_comfyui_audio(wav_vocal[0].cpu().float(), self.sample_rate)
            bgm_audio = tensor_to_comfyui_audio(wav_bgm[0].cpu().float(), self.sample_rate)
        else:
            vocal_audio = empty_audio(self.sample_rate)
            bgm_audio = empty_audio(self.sample_rate)

        print(f"[FL SongGen LowMem] Generation complete!")
        return mixed_audio, vocal_audio, bgm_audio

    def _generate_ultra_lowmem(
        self,
        lyrics: str,
        description: Optional[str],
        prompt_audio: Optional[torch.Tensor],
        auto_style: Optional[str],
        duration: float,
        temperature: float,
        cfg_coef: float,
        top_k: int,
        gen_type: str,
    ) -> Tuple[dict, Optional[dict], Optional[dict]]:
        """
        Ultra low memory generation mode for systems with limited VRAM (~6-8GB).

        Key differences from low_mem mode:
        - Aggressive VRAM cleanup before each phase
        - Loads model weights incrementally
        - Uses CPU for more operations
        - More aggressive garbage collection
        """
        # Import builders for on-demand loading
        from codeclm.models import builders, CodecLM

        cfg = self.config
        ckpt_path = self.model_info["ckpt_path"]

        print(f"[FL SongGen UltraLowMem] Starting generation with aggressive memory management...")

        # Phase 0: Aggressive initial cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Log initial VRAM state
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"[FL SongGen UltraLowMem] Initial VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        # Determine if we need audio tokenizer
        use_audio_tokenizer = prompt_audio is not None

        # Store raw wavs for later
        raw_pmt_wav = None
        raw_vocal_wav = None
        raw_bgm_wav = None

        # Phase 1: Process prompts with audio tokenizer (if needed)
        if use_audio_tokenizer:
            print("[FL SongGen UltraLowMem] Phase 1: Processing audio prompts...")

            # Load and process audio tokenizer
            audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
            audio_tokenizer = audio_tokenizer.eval().cuda()

            # Separate and encode prompt audio
            raw_pmt_wav, raw_vocal_wav, raw_bgm_wav = self._separate_audio(prompt_audio)

            with torch.no_grad():
                pmt_wav, _ = audio_tokenizer.encode(raw_pmt_wav.cuda())

            # Aggressive cleanup
            del audio_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Load separate tokenizer for vocal/bgm
            print("[FL SongGen UltraLowMem] Loading separate tokenizer...")
            separate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
            separate_tokenizer = separate_tokenizer.eval().cuda()

            with torch.no_grad():
                vocal_wav, bgm_wav = separate_tokenizer.encode(
                    raw_vocal_wav.cuda(),
                    raw_bgm_wav.cuda()
                )

            # Aggressive cleanup
            del separate_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            melody_is_wav = False
        else:
            # Use auto prompts or no prompt
            pmt_wav, vocal_wav, bgm_wav, melody_is_wav = self._prepare_prompts(
                None, auto_style, None
            )

        # Phase 2: Generate tokens with LM - ultra careful memory management
        print("[FL SongGen UltraLowMem] Phase 2: Loading language model with memory optimization...")

        # Pre-cleanup before loading large model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"[FL SongGen UltraLowMem] Pre-LM VRAM: {allocated:.2f}GB")

        # Build LM on CPU first
        audiolm = builders.get_lm_model(cfg)

        # Load checkpoint - keep on CPU
        print("[FL SongGen UltraLowMem] Loading checkpoint...")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        audiolm_state_dict = {
            k.replace('audiolm.', ''): v
            for k, v in checkpoint.items()
            if k.startswith('audiolm')
        }

        # Free checkpoint memory immediately
        del checkpoint
        gc.collect()

        # Resize embeddings to match checkpoint
        audiolm = self._resize_embeddings_for_checkpoint(audiolm, audiolm_state_dict)

        # Load state dict on CPU
        audiolm.load_state_dict(audiolm_state_dict, strict=False)

        # Free state dict memory
        del audiolm_state_dict
        gc.collect()

        # Now move to GPU in float16
        print("[FL SongGen UltraLowMem] Moving model to GPU...")
        audiolm = audiolm.eval().to(torch.float16).cuda()

        # Aggressive cleanup after GPU transfer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"[FL SongGen UltraLowMem] Post-LM load VRAM: {allocated:.2f}GB")

        model = CodecLM(
            name="tmp",
            lm=audiolm,
            audiotokenizer=None,
            max_duration=self.max_duration,
            seperate_tokenizer=None,
        )

        model.set_generation_params(
            duration=duration,
            extend_stride=5,
            temperature=temperature,
            cfg_coef=cfg_coef,
            top_k=top_k,
            top_p=0.0,
            record_tokens=True,
            record_window=50
        )

        # Set progress callback
        def progress_wrapper(current, total):
            if self._progress_callback:
                self._progress_callback(current, total)

        model.set_custom_progress_callback(progress_wrapper)

        generate_inp = {
            'lyrics': [lyrics],
            'descriptions': [description],
            'melody_wavs': pmt_wav,
            'vocal_wavs': vocal_wav,
            'bgm_wavs': bgm_wav,
            'melody_is_wav': melody_is_wav,
        }

        print(f"[FL SongGen UltraLowMem] Generating tokens...")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                tokens = model.generate(**generate_inp, return_tokens=True)

        # Phase 2 cleanup - very aggressive
        print("[FL SongGen UltraLowMem] Cleaning up LM...")
        del model

        # Move audiolm to CPU before deletion to free GPU memory faster
        audiolm = audiolm.cpu()
        del audiolm

        # Multiple cleanup rounds
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"[FL SongGen UltraLowMem] Post-LM cleanup VRAM: {allocated:.2f}GB")

        # Phase 3: Decode audio with separate tokenizer
        print("[FL SongGen UltraLowMem] Phase 3: Loading audio decoder...")

        separate_tokenizer = builders.get_audio_tokenizer_model_cpu(cfg.audio_tokenizer_checkpoint_sep, cfg)
        device = "cuda:0"
        separate_tokenizer.model.device = device
        separate_tokenizer.model.vae = separate_tokenizer.model.vae.to(device)
        separate_tokenizer.model.model.device = torch.device(device)
        separate_tokenizer.model.model = separate_tokenizer.model.model.to(device)
        separate_tokenizer = separate_tokenizer.eval()

        model = CodecLM(
            name="tmp",
            lm=None,
            audiotokenizer=None,
            max_duration=self.max_duration,
            seperate_tokenizer=separate_tokenizer,
        )

        print(f"[FL SongGen UltraLowMem] Decoding audio...")
        with torch.no_grad():
            if gen_type == 'separate':
                if raw_pmt_wav is not None:
                    wav_mixed = model.generate_audio(
                        tokens, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav,
                        chunked=True, gen_type='mixed'
                    )
                    wav_vocal = model.generate_audio(
                        tokens, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav,
                        chunked=True, gen_type='vocal'
                    )
                    wav_bgm = model.generate_audio(
                        tokens, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav,
                        chunked=True, gen_type='bgm'
                    )
                else:
                    wav_mixed = model.generate_audio(tokens, chunked=True, gen_type='mixed')
                    wav_vocal = model.generate_audio(tokens, chunked=True, gen_type='vocal')
                    wav_bgm = model.generate_audio(tokens, chunked=True, gen_type='bgm')
            else:
                if raw_pmt_wav is not None:
                    wav_mixed = model.generate_audio(
                        tokens, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav,
                        chunked=True, gen_type=gen_type
                    )
                else:
                    wav_mixed = model.generate_audio(tokens, chunked=True, gen_type=gen_type)
                wav_vocal = None
                wav_bgm = None

        # Final cleanup
        del model
        del separate_tokenizer
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Convert to ComfyUI format
        mixed_audio = tensor_to_comfyui_audio(wav_mixed[0].cpu().float(), self.sample_rate)

        if gen_type == 'separate':
            vocal_audio = tensor_to_comfyui_audio(wav_vocal[0].cpu().float(), self.sample_rate)
            bgm_audio = tensor_to_comfyui_audio(wav_bgm[0].cpu().float(), self.sample_rate)
        else:
            vocal_audio = empty_audio(self.sample_rate)
            bgm_audio = empty_audio(self.sample_rate)

        print(f"[FL SongGen UltraLowMem] Generation complete!")
        return mixed_audio, vocal_audio, bgm_audio

    def _prepare_prompts(
        self,
        prompt_audio: Optional[torch.Tensor],
        auto_style: Optional[str],
        audio_tokenizer: Any
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """
        Prepare prompt tokens for generation.

        Returns:
            (pmt_wav, vocal_wav, bgm_wav, melody_is_wav)
        """
        if prompt_audio is not None and audio_tokenizer is not None:
            # Encode prompt audio
            if prompt_audio.dim() == 2:
                prompt_audio = prompt_audio.unsqueeze(0)
            prompt_audio = prompt_audio.to(self.device)

            with torch.no_grad():
                pmt_wav, _ = audio_tokenizer.encode(prompt_audio)

            return pmt_wav, None, None, False

        elif auto_style is not None and self.auto_prompts is not None:
            # Use auto style prompt
            if auto_style not in self.auto_prompts:
                print(f"[FL SongGen] Warning: Auto style '{auto_style}' not found, using 'Auto'")
                auto_style = "Auto"

            prompt_list = self.auto_prompts[auto_style]
            prompt_token = prompt_list[np.random.randint(0, len(prompt_list))]

            pmt_wav = prompt_token[:, [0], :]
            vocal_wav = prompt_token[:, [1], :]
            bgm_wav = prompt_token[:, [2], :]

            return pmt_wav, vocal_wav, bgm_wav, False

        else:
            # No prompt
            return None, None, None, True

    def _resize_embeddings_for_checkpoint(self, model: torch.nn.Module, state_dict: dict) -> torch.nn.Module:
        """
        Resize embedding layers in the model to match checkpoint dimensions.
        This fixes tokenizer version mismatches where vocab sizes differ.

        Args:
            model: The model to resize embeddings in
            state_dict: The checkpoint state dict to match

        Returns:
            The model with resized embeddings
        """
        import torch.nn as nn

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

        for key, ckpt_tensor in state_dict.items():
            if 'output_proj.weight' in key:
                # Get the path to the parent module (remove .weight)
                module_path = key.rsplit('.', 1)[0]
                current_module = get_nested_attr(model, module_path)

                if current_module is not None and hasattr(current_module, 'weight'):
                    current_size = current_module.weight.shape[0]
                    checkpoint_size = ckpt_tensor.shape[0]

                    if current_size != checkpoint_size:
                        print(f"[FL SongGen LowMem] Resizing embedding {module_path}: {current_size} -> {checkpoint_size}")
                        # Create new embedding with checkpoint size
                        embed_dim = ckpt_tensor.shape[1]
                        padding_idx = getattr(current_module, 'padding_idx', None)
                        new_embedding = nn.Embedding(checkpoint_size, embed_dim, padding_idx=padding_idx)
                        set_nested_attr(model, module_path, new_embedding)

        return model

    def _separate_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Separate audio into full, vocal, and bgm using Demucs.

        Args:
            audio: Audio tensor [B, C, S] at 48kHz

        Returns:
            (full_mix, vocals, bgm) tensors
        """
        # Import from our package explicitly to avoid conflicts
        _model_manager = _import_from_fl_utils("model_manager", "model_manager")
        load_separator = _model_manager.load_separator

        # Ensure 48kHz
        if audio.shape[-1] < 48000 * 10:
            # Pad to 10 seconds
            padding = torch.zeros(audio.shape[0], audio.shape[1], 48000 * 10 - audio.shape[-1])
            audio = torch.cat([audio, padding], dim=-1)
        elif audio.shape[-1] > 48000 * 10:
            audio = audio[..., :48000 * 10]

        # Use a simpler separation: vocal = audio - bgm estimation
        # For proper separation, we'd need Demucs which requires file I/O
        # Let's save to temp file and use the separator

        separator = load_separator(self.device)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            # Use soundfile instead of torchaudio to avoid TorchCodec issues on Windows
            audio_np = audio.squeeze(0).cpu().numpy().T  # (channels, samples) -> (samples, channels)
            sf.write(temp_path, audio_np, 48000)

            # Run separation
            with tempfile.TemporaryDirectory() as tmp_dir:
                separator.separate(temp_path, tmp_dir, device=torch.device(self.device))

                # Get the base name of the temp file (without extension) to find output files
                # The separator names output as {input_basename}_{stem}.flac
                # Note: htdemucs uses 'vocal' (singular), not 'vocals'
                temp_basename = os.path.splitext(os.path.basename(temp_path))[0]
                vocals_path = os.path.join(tmp_dir, f"{temp_basename}_vocal.flac")

                # Load separated audio using soundfile
                vocals_np, sr = sf.read(vocals_path)
                # Convert from (samples, channels) to (channels, samples)
                if vocals_np.ndim == 1:
                    vocals = torch.from_numpy(vocals_np).unsqueeze(0).float()
                else:
                    vocals = torch.from_numpy(vocals_np.T).float()
                if vocals.shape[-1] > 48000 * 10:
                    vocals = vocals[..., :48000 * 10]

            # BGM = full - vocals
            full_audio = audio.squeeze(0)[:, :vocals.shape[-1]]
            bgm = full_audio - vocals

            return (
                audio[:, :, :48000 * 10],
                vocals.unsqueeze(0),
                bgm.unsqueeze(0)
            )

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
