"""
Module for generating transcripts from YouTube video URLs using Whisper and yt_dlp.
"""

import os
import yt_dlp
from faster_whisper import WhisperModel
from pydantic import BaseModel

# Define the input schema
class GenerateTranscriptInput(BaseModel):
    """
    Input model for generating a transcript, which includes:
    - video_url: URL of the YouTube video to process.
    """
    video_url: str

# Environment setting to avoid errors with duplicate libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def generate_transcript(video_url: str) -> dict:
    """Generate a transcript from a YouTube video URL using Whisper."""

    # yt_dlp options for audio extraction
    ydl_opts = {
        'quiet': True,
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    # Download the audio and retrieve metadata
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        video_title = info_dict.get('title', 'Unknown Title')
        video_id = info_dict.get('id', 'unknown')
        video_upload_date = info_dict.get('upload_date', 'unknown')
        audio_file = "temp_audio.mp3"

    # Initialize Whisper model
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    segments = whisper_model.transcribe(audio_file, beam_size=5, word_timestamps=True)[0]

    # Construct transcript from segments
    transcript = []
    transcript_text = ""

    for segment in segments:
        transcript.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
        transcript_text += segment.text + "\n"

    # Clean up temporary audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)

    return {
        "video_id": video_id,
        "title": video_title,
        "upload_date": video_upload_date,
        "transcript": transcript,
        "transcript_text": transcript_text
    }
