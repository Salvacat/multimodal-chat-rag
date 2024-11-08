"""
This module provides functionality to retrieve video URLs from YouTube using the yt-dlp library.
It includes URL validation, cleaning, and extraction of video links from playlists,
channels, or individual videos.
"""

import re  # standard library import first
import yt_dlp  # third-party library
from pydantic import BaseModel

class RetrieveVideoURLsInput(BaseModel):
    """
    Input schema for retrieving video URLs, accepting a single URL as a string.
    """
    input_url: str  # Accept a single URL as a string

def clean_url(url: str) -> str:
    """
    Extracts and returns the main URL from a string.
    Args:
        url (str): URL to clean.
    Returns:
        str: Cleaned URL.
    """
    match = re.search(r'https?://[^\s]+', url)
    return match.group(0) if match else url

def retrieve_video_urls(input_url: str) -> list:
    """
    Retrieve video URLs from a YouTube playlist, channel, or individual video URL.
    Args:
        input_url (str): A single URL to process.
    Returns:
        list: Retrieved video URLs.
    """
    # Clean the URL before passing it to yt-dlp
    input_url = clean_url(input_url)

    ydl_opts = {
        'quiet': True,
        'extract_flat': 'in_playlist',
        'format': 'best',
    }

    retrieved_urls = []
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(input_url, download=False)
            # Check if info_dict has 'entries' (multiple videos), or just one video URL
            if 'entries' in info_dict:
                video_urls = [entry['url'] for entry in info_dict['entries'] if 'url' in entry]
            else:
                video_urls = [info_dict['webpage_url']]
            retrieved_urls.extend(video_urls)

    except yt_dlp.utils.DownloadError as e:
        print(f"Error retrieving URL {input_url}: {e}")

    return retrieved_urls