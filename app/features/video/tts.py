"""Piper TTS wrapper — local, free neural text-to-speech."""

import os
import wave
from functools import lru_cache

from piper import PiperVoice


@lru_cache(maxsize=1)
def get_voice() -> PiperVoice:
    voice_path = os.getenv("PIPER_VOICE_PATH", "voices/en_US-lessac-medium.onnx")
    return PiperVoice.load(voice_path)


def synthesize_to_wav(text: str, output_path: str) -> None:
    voice = get_voice()
    with wave.open(output_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
