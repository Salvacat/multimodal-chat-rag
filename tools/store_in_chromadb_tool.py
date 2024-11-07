"""
This module provides functionality to store content in ChromaDB with metadata.
The content is split into chunks for efficient storage and retrieval in the database.
"""

from typing import Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

class StoreInChromaDBInput(BaseModel):
    """
    Input schema for storing content in ChromaDB, including content,
    metadata, and a reference to the vector database instance.
    """
    content: str = Field(..., description="The content to store in ChromaDB.")
    metadata: Dict[str, Any] = Field(..., description="Metadata associated with the content.")
    vector_db: Any = Field(..., description="The vector database instance.")

# Initialize text splitter with chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

def store_in_chromadb(content: str, metadata: Dict[str, Any], vector_db: Any) -> None:
    """
    Store content in ChromaDB with metadata, using text splitting for large content.
    Args:
        content (str): The main text content to be stored.
        metadata (Dict[str, Any]): Metadata to associate with each chunk of content.
        vector_db (Any): Instance of the vector database for storing the text.
    """
    # Split content into chunks
    split_content = text_splitter.split_text(content)

    # Add each chunk to the database with the same metadata
    for i, chunk in enumerate(split_content):
        chunk_metadata = metadata.copy()  # Copy metadata for each chunk
        chunk_metadata["chunk_index"] = i  # Add an index to each chunk for tracking

        # Store the chunk with metadata in ChromaDB
        vector_db.add_texts(texts=[chunk], metadatas=[chunk_metadata])
        print(f"Stored chunk {i + 1}/{len(split_content)} for {metadata['source']} in ChromaDB.")
