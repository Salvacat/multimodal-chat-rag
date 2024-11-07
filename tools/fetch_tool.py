"""
Tool for fetching and formatting YouTube video transcripts with configurable chunking.
"""

from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_community.document_loaders import YoutubeLoader
from pydantic import BaseModel

# Define the input schema
class FetchTranscriptToolInput(BaseModel):
    """
    Schema for fetching a transcript from YouTube, including:
    - video_url: URL of the YouTube video.
    - chunk_size_seconds: Size of chunks in seconds, defaulting to 30.
    """
    video_url: str
    chunk_size_seconds: int = 30  # Default value if not provided

def fetch_transcript_tool(video_url: str, chunk_size_seconds: int = 30) -> str:
    """
    Retrieve a transcript from YouTube with artificial timestamps if needed.
    
    Args:
        video_url (str): URL of the YouTube video.
        chunk_size_seconds (int): Duration in seconds for each transcript chunk.

    Returns:
        str: Formatted transcript text with timestamps or None if an error occurs.
    """
    try:
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=False,
            language=["en"],
            translation=None,
            transcript_format=(
                TranscriptFormat.CHUNKS if chunk_size_seconds else TranscriptFormat.TEXT
            ),
            chunk_size_seconds=chunk_size_seconds
        )
        transcript_docs = loader.load()

        transcript_text = ""
        for i, doc in enumerate(transcript_docs):
            start_time = i * chunk_size_seconds
            end_time = (i + 1) * chunk_size_seconds
            timestamp = (
                f"{start_time // 60:02}:{start_time % 60:02}--"
                f"{end_time // 60:02}:{end_time % 60:02}"
            )
            transcript_text += f"{timestamp}: {doc.page_content}\n"

        return transcript_text
    except Exception as e:
        print(f"Error retrieving transcript: {e}")
        return None
