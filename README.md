+++
# Project Title: InsightVault

## Description
InsightVault is a modular tool designed to fetch or generate transcripts for YouTube videos, playlists, and channels, storing the transcripts in ChromaDB for efficient retrieval. Users can interact with stored transcripts through a question-answering agent, enabling quick and relevant responses based on video content. If transcripts are unavailable directly from YouTube, the system generates them using Whisper. The design allows for seamless integration of additional tools and LLMs, making it a versatile foundation for future expansions.

---

## Project Structure

The project is organized as follows:

- **`app.py`**: Launches the Gradio interface for user interaction.
- **`config.py`**: Stores configuration variables, including API keys and database paths.
- **`environment.env`**: Holds sensitive information, such as API keys.
- **`evaluation.py`**: Script to evaluate agent responses against a custom dataset.
- **`main.py`**: Sets up the agent, tools, and vector database.
- **`memory.py`**: Manages conversation memory to enable contextual responses.
- **`retrievers.py`**: Contains functions for querying ChromaDB.
- **`tools/`**: Directory containing modular tools, each designed for a specific task:
    - **`fetch_tool.py`**: Fetches transcripts from YouTube if available.
    - **`generate_transcript_tool.py`**: Generates transcripts with Whisper when necessary.
    - **`retrieve_or_generate_tool.py`**: Logic to either retrieve or generate transcripts and store them in ChromaDB.
    - **`store_in_chromadb_tool.py`**: Stores processed transcripts in ChromaDB.
    - **`video_urls_tool.py`**: Validates and retrieves video URLs from playlists, channels, or single videos.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment setup**
   Configure an `.env` file with necessary API keys and paths:
   ```plaintext
   OPENAI_API_KEY=your_openai_key
   CHROMA_DB_PATH=path_to_chromadb
   ```

---

## Usage

To start the Gradio app, run:

```bash
python app.py
```

Once launched, the interface supports:

- **YouTube URL Submission**: Provide a URL for a single video, playlist, or channel to fetch or generate transcripts and store them in ChromaDB. The system can handle entire playlists or all videos from a channel by automatically extracting and processing each video URL.
- **Querying Stored Transcripts**: Use the Gradio interface to ask questions and retrieve relevant information from stored transcripts.
- **Error Handling**: If no answer is provided, try rephrasing the question or clearing memory.
- **Memory Reset**: Use the reset option in Gradio to clear conversation history for a fresh session.

---

## Evaluation

The `evaluation.py` script evaluates agent responses against a pre-defined dataset in LangSmith.

### Evaluation Setup
- **Question-Answer Matching**: Uses a custom dataset with sample questions and expected answers for comparison.
- **Document Relevance and Answer Quality**: Utilizes custom evaluators to assess if the retrieved documents and responses are relevant and accurate.
- **Creating a Custom Dataset**: Users can create a tailored dataset in LangSmith with questions and expected answers for custom evaluation.

---

## Known Limitations

- **GPU Dependency Warnings**: You may encounter warnings related to CUDA or other GPU dependencies. These are usually safe to ignore unless GPU processing is required.
- **Pydantic Validation Errors**: Ensure proper formatting of metadata and database parameters for ChromaDB interactions.

---

## Dependencies

- **LangChain**: For managing the agent and tool pipeline.
- **Gradio**: For a web interface enabling easy user interaction.
- **ChromaDB**: Vector database for storing and retrieving transcripts.
- **Whisper**: Used for transcript generation when not available from YouTube.
+++