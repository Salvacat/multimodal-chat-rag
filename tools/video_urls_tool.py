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

def is_valid_url(url: str) -> bool:
    """
    Validate if a given URL is properly formatted.
    Args:
        url (str): URL to validate.
    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    url_regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    return re.match(url_regex, url) is not None

def clean_url(url: str) -> str:
    """
    Extracts and returns the main URL from a string.
    Args:
        url (str): URL to clean.
    Returns:
        str: Cleaned URL.
    """
    match = re.search(r'https?://\S+', url)
    return match.group(0) if match else url

def retrieve_video_urls(input_url) -> list:
    """
    Retrieve video URLs from a YouTube playlist, channel, or individual video URL.
    Args:
        input_url (str or list): A single URL or list of URLs to process.
    Returns:
        list: Retrieved video URLs.
    """
    input_urls = [input_url] if isinstance(input_url, str) else input_url
    failed_urls = []
    retrieved_urls = []

    for url in input_urls:
        url = clean_url(url)
        if not is_valid_url(url):
            print(f"Invalid URL provided: {url}")
            failed_urls.append(url)
            continue

        ydl_opts = {
            'quiet': True,
            'extract_flat': 'in_playlist',
            'format': 'best',
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                video_urls = [
                    entry['url'] for entry in info_dict.get('entries', [])
                    if 'url' in entry
                ] if 'entries' in info_dict else [info_dict.get('webpage_url')]
                retrieved_urls.extend(video_urls)
        except yt_dlp.utils.DownloadError as e:
            print(f"Error retrieving URL {url}: {e}")
            failed_urls.append(url)

    if failed_urls:
        print("Failed to retrieve the following URLs:")
        for url in failed_urls:
            print(f"- {url}")

    return retrieved_urls
