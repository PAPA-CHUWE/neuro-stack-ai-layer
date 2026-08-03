"""ffmpeg orchestration — renders each slide+narration as a clip, then concatenates."""

import os
import subprocess
import tempfile

from app.features.video.slides import render_slide
from app.features.video.tts import synthesize_to_wav


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")


def build_video(slides: list[dict]) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths: list[str] = []

        for i, slide in enumerate(slides):
            img = render_slide(slide.get("title", ""), slide.get("bullets", []))
            img_path = os.path.join(tmpdir, f"slide_{i}.png")
            img.save(img_path)

            wav_path = os.path.join(tmpdir, f"slide_{i}.wav")
            synthesize_to_wav(slide.get("narration") or slide.get("title", ""), wav_path)

            clip_path = os.path.join(tmpdir, f"clip_{i}.mp4")
            _run_ffmpeg([
                "-loop", "1", "-i", img_path,
                "-i", wav_path,
                "-c:v", "libx264", "-tune", "stillimage",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p", "-shortest",
                clip_path,
            ])
            clip_paths.append(clip_path)

        concat_list_path = os.path.join(tmpdir, "concat.txt")
        with open(concat_list_path, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")

        final_path = os.path.join(tmpdir, "final.mp4")
        _run_ffmpeg([
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path, "-c", "copy", final_path,
        ])

        with open(final_path, "rb") as f:
            return f.read()
