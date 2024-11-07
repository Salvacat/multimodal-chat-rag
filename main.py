"""
This module provides a structured pipeline for processing YouTube video data.
It retrieves video URLs, generates transcripts, stores information in a ChromaDB,
and retrieves data using an LLM agent with conversation memory.
"""

# Imports
from pydantic import BaseModel
from langchain.output_parsers import RegexDictParser
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Internal tool imports
from tools.video_urls_tool import retrieve_video_urls, RetrieveVideoURLsInput
from tools.generate_transcript_tool import generate_transcript, GenerateTranscriptInput
from tools.store_in_chromadb_tool import store_in_chromadb, StoreInChromaDBInput
from tools.retrieve_or_generate_tool import (
    retrieve_or_generate_transcripts,
    RetrieveOrGenerateTranscriptsInput
)
from tools.fetch_tool import fetch_transcript_tool, FetchTranscriptToolInput
from retrievers import multiquery_wrapper, MultiQueryInput
from memory import ConversationMemoryRunnable
from config import OPENAI_API_KEY, CHROMA_DB_PATH

# Initialize the Hugging Face embedding model
EMBEDDING_FUNCTION = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vector_db
VECTOR_DB = Chroma(
    collection_name="multimedia_documents",
    embedding_function=EMBEDDING_FUNCTION,
    persist_directory=CHROMA_DB_PATH,
)

# Define tools
video_urls_tool = StructuredTool(
    name="retrieve_video_urls",
    func=retrieve_video_urls,
    description="Retrieve video URLs from a list of YouTube playlists, channels, or single videos.",
    args_schema=RetrieveVideoURLsInput
)

fetch_tool = StructuredTool(
    name="fetch_transcript_tool",
    func=fetch_transcript_tool,
    description=("Fetches the transcript of a YouTube "
                 "video based on the provided URL and chunk size."),
    args_schema=FetchTranscriptToolInput
)

generate_transcript_tool = StructuredTool(
    name="generate_transcript",
    func=generate_transcript,
    description="Generate a transcript from a YouTube video URL using Whisper.",
    args_schema=GenerateTranscriptInput
)

store_in_chromadb_tool = StructuredTool(
    name="store_in_chromadb",
    func=store_in_chromadb,
    description="Store content in ChromaDB with metadata, using text splitting for large content.",
    args_schema=StoreInChromaDBInput
)

retrieve_or_generate_tool = StructuredTool(
    name="retrieve_or_generate_transcript",
    func=lambda video_urls, vector_db: retrieve_or_generate_transcripts(
        video_urls, vector_db, store_in_chromadb_tool
    ),
    description=(
        "Retrieve transcript from YouTube or generate using Whisper if unavailable "
        "and store it in ChromaDB."
    ),
    args_schema=RetrieveOrGenerateTranscriptsInput
)

multiquery_tool = StructuredTool(
    name="multiquery",
    func=lambda query: multiquery_wrapper(query, llm=llm, vector_db=VECTOR_DB),
    description="Retrieve documents discussing '{query}' with a similarity threshold filter.",
    args_schema=MultiQueryInput
)

# Define pipeline for document retrieval and storage
pipeline = (
    video_urls_tool | (lambda urls: {"video_urls": urls, "vector_db": VECTOR_DB}) |
    retrieve_or_generate_tool
)

class PipelineInput(BaseModel):
    """Schema for pipeline input."""
    input_url: str  # Initial video or playlist URL for `video_urls_tool`

pipeline_tool = StructuredTool(
    name="document_retrieval_pipeline",
    func=pipeline.invoke,  # Directly using pipeline.invoke
    description="Retrieves new documents and stores relevant information in the database.",
    args_schema=PipelineInput,
)

# Initialize the LLM with memory functionality
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
memory_llm = ConversationMemoryRunnable(llm)

# Combine tools with LLM and memory
all_tools = [
    pipeline_tool,
    store_in_chromadb_tool,
    multiquery_tool
]

# Define regex patterns for agent parsing
patterns = {
    "question": r"Question:\s*(.*)",
    "thought": r"Thought:\s*(.*)",
    "action": r"Action:\s*(.*)",
    "action_input": r"Action Input:\s*(.*)",
    "observation": r"Observation:\s*(.*)",
    "final_answer": r"Final Answer:\s*(.*)"
}

output_parser = RegexDictParser(
    patterns=patterns,
    output_key_to_format={"final_answer": "Final Answer"}
)

# Define tool descriptions and names
TOOL_DESCRIPTIONS = "\n".join(
    [f"{tool.name}: {tool.description}" for tool in all_tools]
)
TOOL_NAMES = ", ".join([tool.name for tool in all_tools])

# Define prompt template
PROMPT_TEMPLATE_TEXT = '''
You are a thoughtful assistant with access to the following tools: {tools}.

When answering questions, **prioritize information from the vector store (RAG)**. 
If relevant content is not found in the vector store **and 
the question explicitly asks for retrieval or more details**, use other tools as directed.

**Important Note**:
- If you cannot find relevant information and no explicit retrieval is requested, respond with "I'm sorry, I don't have information on that."
- Always follow the specified format strictly.

Use the following format for every interaction:
python app.py        
Question: {input}
Thought: think of an action based on the question (Do not include any additional information, metadata, token usage, or other details.
 or "I'm sorry" responses if a retrieval is possible)
Action: select one [{tool_names}]
Action Input: the input to the action. Do not include any additional information, metadata, token usage, or other details.
Observation: the result of the action
...(repeat Thought/Action/Observation as needed)
Only provide "Final Answer:" if you have completed the analysis. Otherwise, continue with "Thought:", "Action:", and "Action Input:" as needed.
Thought: I now know the final answer

"Please respond with only the final answer in this exact format:
Final Answer: [Your answer here]
Do not include any additional information, metadata, token usage, or other details."

Begin!

Question: {input}
Thought: {agent_scratchpad}
'''

# Create the prompt template
prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE_TEXT)

# Initialize the agent
agent = create_react_agent(
    llm=memory_llm,
    tools=all_tools,
    prompt=prompt_template
)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True,
    handle_parsing_errors=True,
    output_parser=output_parser
)
