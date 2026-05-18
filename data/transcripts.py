from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import subprocess
import os
import tempfile
from config.config import CONFIG

def extract_video_id(youtube_url: str) -> str:
    parsed_url = urlparse(youtube_url)

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query)["v"][0]

        if parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/")[2]

    raise ValueError("Invalid YouTube URL")

def normalize_transcript(transcript):
    if hasattr(transcript, "snippets"):
        return " ".join([s.text for s in transcript.snippets])

    if isinstance(transcript, list):
        return " ".join([
            getattr(t, "text", t.get("text", "")) for t in transcript
        ])

    if isinstance(transcript, dict):
        return transcript.get("text", "")

    if isinstance(transcript, str):
        return transcript

    raise ValueError(f"Unsupported transcript format: {type(transcript)}")

def fetch_from_api(video_id, language="en"):
    yt_api = YouTubeTranscriptApi()
    raw = yt_api.fetch(video_id, languages=[language])
    return normalize_transcript(raw)


def fetch_with_whisper(youtube_url, language="en"):
    """
    Requires:
    - yt-dlp installed
    - whisper installed (open-source)
    """
    if not CONFIG.get("WHISPER_ENABLED", False):
        raise RuntimeError("Whisper fallback is disabled")

    model = CONFIG.get("WHISPER_MODEL", "base")
    whisper_lang = CONFIG.get("WHISPER_LANGUAGE", language)

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_file = os.path.join(temp_dir, "audio.mp3")
        txt_file = os.path.join(temp_dir, "audio.txt")

        try:
            subprocess.run([
                "yt-dlp",
                "-x",
                "--audio-format", "mp3",
                "-o", audio_file,
                youtube_url
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"yt-dlp download failed: {e.stderr}")

        try:
            result = subprocess.run([
                "whisper",
                audio_file,
                "--model", model,
                "--language", whisper_lang,
                "--output_format", "txt",
                "--output_dir", temp_dir
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Whisper transcription failed: {e.stderr}")

        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                return f.read().strip()

        raise RuntimeError("Whisper did not produce output file")

def get_transcript(youtube_url: str, language="en"):
    video_id = extract_video_id(youtube_url)

    try:
        text = fetch_from_api(video_id, language)

        return {
            "video_id": video_id,
            "source": "youtube_transcript_api",
            "text": text
        }

    except Exception as e:
        print(f"[WARN] Primary transcript failed: {e}")
        print("[INFO] Switching to Whisper fallback...")

        try:
            text = fetch_with_whisper(youtube_url, language)

            return {
                "video_id": video_id,
                "source": "whisper_fallback",
                "text": text
            }

        except Exception as e2:
            return {
                "video_id": video_id,
                "error": f"Both methods failed: {str(e2)}"
            }