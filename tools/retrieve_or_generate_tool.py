"""
This module provides functionality to retrieve or generate transcripts
for a list of YouTube video URLs, storing them in ChromaDB.
"""

from typing import List, Any
from pydantic import BaseModel
from tools.generate_transcript_tool import generate_transcript  # pylint: disable=E0401
from tools.fetch_tool import fetch_transcript_tool  # pylint: disable=E0401

class RetrieveOrGenerateTranscriptsInput(BaseModel):
    """
    Input model for retrieving or generating transcripts. Includes:
    - video_urls: List of YouTube video URLs to process.
    - vector_db: Database instance for storing transcript data.
    """
    video_urls: List[str]
    vector_db: Any

def retrieve_or_generate_transcripts(
    video_urls: List[str],
    vector_db: Any,
    store_in_chromadb_tool
) -> str:
    """
    Retrieve or generate transcripts for each video URL, store them in ChromaDB,
    and return a summary of the storage status.

    Args:
        video_urls (List[str]): List of YouTube video URLs.
        vector_db (Any): The vector database instance for storage.
        store_in_chromadb_tool: Tool to handle storage in ChromaDB.

    Returns:
        str: Confirmation summary of transcripts stored.
    """
    stored_urls = []

    for url in video_urls:
        try:
            # Attempt to fetch or generate the transcript
            transcript_text = fetch_transcript_tool(url)
            if transcript_text is None:
                generated_data = generate_transcript(url)
                transcript_text = generated_data.get("transcript_text")
                print(f"Generated transcript for {url}.")
            else:
                print(f"Fetched transcript for {url} from YouTube.")

            # Store the transcript in ChromaDB if available
            if transcript_text:
                metadata = {
                    "source": "YouTube",
                    "video_url": url,
                    "document_type": "transcript"
                }
                tool_input = {
                    "content": transcript_text,
                    "metadata": metadata,
                    "vector_db": vector_db
                }

                # Ensure tool_input matches StoreInChromaDBInput's structure
                store_in_chromadb_tool.invoke(tool_input)
                stored_urls.append(url)
                print(f"Transcript for {url} stored in ChromaDB.")
            else:
                print(f"No transcript available for {url}.")

        except Exception as e:  # Catch any unexpected errors
            print(f"Error retrieving or generating transcript for {url}: {e}")

    return f"Transcripts created and stored for {len(stored_urls)} videos."
