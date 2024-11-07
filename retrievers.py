"""
This module defines a multiquery tool for retrieving relevant documents using
a similarity threshold filter, leveraging LangChain's MultiQueryRetriever.
"""

from langchain.retrievers import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

# Define the input schema for the multiquery tool
class MultiQueryInput(BaseModel):
    """
    Defines the input schema for the multiquery tool, including the query
    and similarity threshold.
    """
    query: str
    similarity_threshold: float = 0.75  # Default threshold; adjustable

# Wrapper function with a similarity threshold filter
def multiquery_wrapper(query: str, llm, vector_db,
                       similarity_threshold: float = 0.75) -> str:
    """
    Retrieves and filters documents based on the specified similarity threshold.

    Args:
        query (str): The user query for which documents are retrieved.
        llm: The language model to use for retrieval.
        vector_db: The vector database configured as a retriever.
        similarity_threshold (float, optional): Minimum similarity score for a document
        to be included.

    Returns:
        str: Combined content of relevant documents that meet the similarity threshold.
    """
    # Define the prompt with improved instruction specificity
    multiquery_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "Retrieve the most relevant documents for: {question}. "
            "Ensure the documents specifically discuss the topic without unrelated content."
        )
    )

    # Configure the retriever with additional parameters for fine-tuning
    retriever = MultiQueryRetriever.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        prompt=multiquery_prompt
    )

    # Retrieve documents
    documents = retriever.invoke({"question": query})

    # Filter documents based on similarity score
    filtered_documents = [
        doc for doc in documents
        if doc.metadata.get("similarity_score", 1.0) >= similarity_threshold
    ]

    # Combine the filtered documents
    text_content = "\n\n".join([doc.page_content for doc in filtered_documents])
    return text_content
