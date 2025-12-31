"""
FL Song Gen Lyrics Formatter Node.
Helps users build properly formatted lyrics with section tags.
"""

from typing import Tuple


class FL_SongGen_LyricsFormatter:
    """
    Build properly formatted lyrics from song sections.

    This helper node makes it easy to construct lyrics in the format
    required by SongGeneration without knowing the exact syntax.

    Format: Sections separated by " ; "
    Lyrics within sections separated by "."
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_lyrics",)
    FUNCTION = "format_lyrics"
    CATEGORY = "FL Song Gen"

    # Section tags available in the model
    INTRO_TYPES = ["none", "intro-short", "intro-medium", "intro-long"]
    OUTRO_TYPES = ["none", "outro-short", "outro-medium", "outro-long"]
    INST_TYPES = ["none", "inst-short", "inst-medium", "inst-long"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "intro_type": (
                    cls.INTRO_TYPES,
                    {
                        "default": "intro-short",
                        "tooltip": "Instrumental intro duration"
                    }
                ),
                "outro_type": (
                    cls.OUTRO_TYPES,
                    {
                        "default": "outro-short",
                        "tooltip": "Instrumental outro duration"
                    }
                ),
            },
            "optional": {
                "verse_1": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "First verse lyrics. Each line becomes a phrase."
                    }
                ),
                "chorus_1": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "First chorus lyrics."
                    }
                ),
                "instrumental_1": (
                    cls.INST_TYPES,
                    {
                        "default": "none",
                        "tooltip": "Instrumental break after first chorus"
                    }
                ),
                "verse_2": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Second verse lyrics."
                    }
                ),
                "chorus_2": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Second chorus lyrics."
                    }
                ),
                "bridge": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Bridge section lyrics."
                    }
                ),
                "instrumental_2": (
                    cls.INST_TYPES,
                    {
                        "default": "none",
                        "tooltip": "Instrumental break after bridge"
                    }
                ),
                "verse_3": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Third verse lyrics (optional)."
                    }
                ),
                "chorus_3": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Final chorus lyrics."
                    }
                ),
                "raw_lyrics": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "OR: Provide pre-formatted lyrics directly (overrides above fields)"
                    }
                ),
            }
        }

    def format_lyrics(
        self,
        intro_type: str,
        outro_type: str,
        verse_1: str = "",
        chorus_1: str = "",
        instrumental_1: str = "none",
        verse_2: str = "",
        chorus_2: str = "",
        bridge: str = "",
        instrumental_2: str = "none",
        verse_3: str = "",
        chorus_3: str = "",
        raw_lyrics: str = "",
    ) -> Tuple[str]:
        """
        Combine sections into formatted lyrics string.

        Args:
            intro_type: Type of intro (none, intro-short, etc.)
            outro_type: Type of outro
            verse_1-3: Verse lyrics
            chorus_1-3: Chorus lyrics
            bridge: Bridge lyrics
            instrumental_1-2: Instrumental break types
            raw_lyrics: Pre-formatted lyrics (overrides other fields)

        Returns:
            Formatted lyrics string ready for generation
        """
        # If raw lyrics provided, use those
        if raw_lyrics.strip():
            print(f"[FL SongGen Lyrics] Using raw lyrics input")
            return (raw_lyrics.strip(),)

        sections = []

        # Intro
        if intro_type != "none":
            sections.append(f"[{intro_type}]")

        # Verse 1
        if verse_1.strip():
            formatted = self._format_section(verse_1)
            sections.append(f"[verse] {formatted}")

        # Chorus 1
        if chorus_1.strip():
            formatted = self._format_section(chorus_1)
            sections.append(f"[chorus] {formatted}")

        # Instrumental 1
        if instrumental_1 != "none":
            sections.append(f"[{instrumental_1}]")

        # Verse 2
        if verse_2.strip():
            formatted = self._format_section(verse_2)
            sections.append(f"[verse] {formatted}")

        # Chorus 2
        if chorus_2.strip():
            formatted = self._format_section(chorus_2)
            sections.append(f"[chorus] {formatted}")

        # Bridge
        if bridge.strip():
            formatted = self._format_section(bridge)
            sections.append(f"[bridge] {formatted}")

        # Instrumental 2
        if instrumental_2 != "none":
            sections.append(f"[{instrumental_2}]")

        # Verse 3
        if verse_3.strip():
            formatted = self._format_section(verse_3)
            sections.append(f"[verse] {formatted}")

        # Chorus 3
        if chorus_3.strip():
            formatted = self._format_section(chorus_3)
            sections.append(f"[chorus] {formatted}")

        # Outro
        if outro_type != "none":
            sections.append(f"[{outro_type}]")

        # Join with section separator
        result = " ; ".join(sections)

        print(f"[FL SongGen Lyrics] Formatted {len(sections)} sections")
        print(f"[FL SongGen Lyrics] Preview: {result[:100]}...")

        return (result,)

    def _format_section(self, text: str) -> str:
        """
        Convert multiline text to SongGen format.
        Newlines become periods to separate phrases.
        """
        # Split by newlines, strip each line, filter empty
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

        # Join with period-space (matches original SongGeneration training data format)
        return ". ".join(lines)
