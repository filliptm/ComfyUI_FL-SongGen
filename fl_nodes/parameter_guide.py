"""
FL Song Gen Parameter Guide Node.
Displays recommended settings for each model variant.
"""

from typing import Tuple


# Model-specific parameter recommendations
_PARAM_GUIDE = {
    "songgeneration_v2_large": {
        "name": "v2-large",
        "temperature": "0.8 (official default)",
        "cfg_coef": "1.5 (official default)",
        "top_k": "5000 (official default)",
        "max_duration": "270s (4m30s)",
        "languages": "Chinese, English, Spanish, Japanese, Korean, French, German, Portuguese, Italian, Russian",
        "vram": "28GB normal / 22GB low VRAM",
        "notes": "Best quality. Multilingual. [Musicality-very-high] tag auto-prepended to description.",
    },
    "songgeneration_v1_5_beta": {
        "name": "v1.5-beta",
        "temperature": "0.9",
        "cfg_coef": "1.5",
        "top_k": "50",
        "max_duration": "270s (4m30s)",
        "languages": "Chinese, English, Spanish, Japanese",
        "vram": "24GB normal / 16GB low VRAM",
        "notes": "Experimental multilingual. May have quirks.",
    },
    "songgeneration_large": {
        "name": "large (v1)",
        "temperature": "0.9",
        "cfg_coef": "1.5",
        "top_k": "50",
        "max_duration": "270s (4m30s)",
        "languages": "Chinese, English",
        "vram": "28GB normal / 22GB low VRAM",
        "notes": "Original large model. Good quality, bilingual.",
    },
    "songgeneration_base_full": {
        "name": "base-full",
        "temperature": "0.9",
        "cfg_coef": "1.5",
        "top_k": "50",
        "max_duration": "270s (4m30s)",
        "languages": "Chinese, English",
        "vram": "18GB normal / 12GB low VRAM",
        "notes": "Extended base model. Bilingual, longer duration.",
    },
    "songgeneration_base_new": {
        "name": "base-new",
        "temperature": "0.9",
        "cfg_coef": "1.5",
        "top_k": "50",
        "max_duration": "150s (2m30s)",
        "languages": "Chinese, English",
        "vram": "16GB normal / 10GB low VRAM",
        "notes": "Lightweight bilingual model. Good for testing.",
    },
    "songgeneration_base": {
        "name": "base",
        "temperature": "0.9",
        "cfg_coef": "1.5",
        "top_k": "50",
        "max_duration": "150s (2m30s)",
        "languages": "Chinese only",
        "vram": "16GB normal / 10GB low VRAM",
        "notes": "Original base model. Chinese only.",
    },
}

_GENERAL_TIPS = """
--- General Tips ---
- Lyrics use section tags: [intro], [verse], [chorus], [bridge], [outro], [intro-short], [outro-short]
- Separate lines with periods (.) or semicolons (;) within sections
- Description controls style: "female, pop, emotional, piano and drums, the bpm is 120"
- Gen type 'Separate All' gives you individual vocal + BGM tracks
- Seed fixes randomness for reproducible results (-1 = random)

--- Parameter Effects ---
- Temperature: creativity dial. 0.5 = very consistent, 1.0 = balanced, 1.5+ = wild
- CFG (guidance): how closely to follow lyrics/description. 1.0 = loose, 3.0+ = strict
- Top-K: diversity of token choices. 50 = safe, 500+ = experimental
"""


class FL_SongGen_ParameterGuide:
    """
    Display recommended parameters for the selected model variant.

    Place this node on your canvas as a reference card. Connect the model
    to see variant-specific recommendations, or leave disconnected for
    general tips.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("guide_text",)
    FUNCTION = "get_guide"
    CATEGORY = "FL Song Gen"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model": (
                    "SONGGEN_MODEL",
                    {
                        "tooltip": "Connect model to see variant-specific recommendations"
                    }
                ),
            }
        }

    def get_guide(self, model: dict = None) -> Tuple[str]:
        if model is not None:
            variant = model.get("variant", "")
            guide = _PARAM_GUIDE.get(variant, {})
            if guide:
                lines = [
                    f"=== {guide['name']} Recommended Settings ===",
                    f"",
                    f"Temperature:  {guide['temperature']}",
                    f"CFG:          {guide['cfg_coef']}",
                    f"Top-K:        {guide['top_k']}",
                    f"Max Duration: {guide['max_duration']}",
                    f"Languages:    {guide['languages']}",
                    f"VRAM:         {guide['vram']}",
                    f"",
                    f"Notes: {guide['notes']}",
                    f"",
                    _GENERAL_TIPS,
                ]
                text = "\n".join(lines)
            else:
                text = f"Unknown variant: {variant}\n" + _GENERAL_TIPS
        else:
            text = "=== FL Song Gen Parameter Guide ===\n\nConnect a model to see variant-specific recommendations.\n" + _GENERAL_TIPS

        print(f"[FL SongGen] Parameter guide displayed")
        return {"ui": {"text": [text]}, "result": (text,)}
