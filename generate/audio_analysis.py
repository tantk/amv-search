"""Analyze audio using CLAP to generate text descriptions of each segment.

CLAP (Contrastive Language-Audio Pretraining) embeds audio and text into
the same vector space. We embed audio segments and find the closest text
descriptions from a candidate pool. These descriptions are then combined
with lyrics to create richer search queries for turbopuffer.
"""

import numpy as np
import torch
import librosa

_clap_model = None
_clap_processor = None

# Candidate mood/scene descriptions that CLAP matches audio against
MOOD_CANDIDATES = [
    "calm peaceful nature",
    "dark moody atmosphere",
    "bright energetic action",
    "epic dramatic cinematic",
    "quiet gentle soft ambient",
    "loud intense powerful",
    "soaring flying through sky",
    "fire flames burning intense",
    "ocean waves water flowing",
    "mountain peaks landscape",
    "stars night sky galaxy space",
    "sunrise golden light warm",
    "storm thunder dark clouds",
    "city lights urban nightlife",
    "forest trees nature green",
    "rain water melancholy",
    "romantic warm tender",
    "triumphant victory celebration",
    "mysterious foggy eerie",
    "joyful happy playful",
]

_text_embeds = None


def _get_clap():
    global _clap_model, _clap_processor, _text_embeds
    if _clap_model is None:
        from transformers import ClapModel, ClapProcessor
        _clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        _clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

        # Pre-embed candidate texts
        text_inputs = _clap_processor(text=MOOD_CANDIDATES, return_tensors="pt", padding=True)
        with torch.no_grad():
            _text_embeds = _clap_model.get_text_features(**text_inputs).pooler_output
            _text_embeds = _text_embeds / _text_embeds.norm(dim=-1, keepdim=True)

    return _clap_model, _clap_processor, _text_embeds


def analyze_audio(audio_path: str, segment_dur: float = 5.0) -> list[dict]:
    """Analyze audio and return CLAP text descriptions per segment.

    Returns list of {"start_ms", "end_ms", "description"} dicts.
    The description is the top-matching mood text for that audio segment.
    """
    model, processor, text_embeds = _get_clap()

    y, sr = librosa.load(audio_path, sr=48000)
    duration = len(y) / sr

    results = []
    n_segments = int(duration / segment_dur)

    for i in range(n_segments + 1):
        start_s = i * segment_dur
        end_s = min((i + 1) * segment_dur, duration)
        if end_s - start_s < 1.0:
            break

        segment = y[int(start_s * 48000):int(end_s * 48000)]

        audio_inputs = processor(audio=segment, sampling_rate=48000, return_tensors="pt")
        with torch.no_grad():
            audio_embed = model.get_audio_features(**audio_inputs).pooler_output
            audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)

        sims = (audio_embed @ text_embeds.T).squeeze()
        top_idx = sims.argsort(descending=True)[:2]
        description = ", ".join(MOOD_CANDIDATES[j] for j in top_idx)

        results.append({
            "start_ms": int(start_s * 1000),
            "end_ms": int(end_s * 1000),
            "description": description,
        })

    return results


def get_mood_for_timestamp(analysis: list[dict], timestamp_ms: int) -> str:
    """Look up the CLAP mood description for a given timestamp."""
    for seg in analysis:
        if seg["start_ms"] <= timestamp_ms < seg["end_ms"]:
            return seg["description"]
    return ""
