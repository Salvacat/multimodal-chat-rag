"""
This module handles agent evaluation using LangSmith, including token overlap
and document relevance assessments for various evaluation metrics.
"""

import re
import time
from collections import Counter
from langsmith import Client
from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY
from main import agent_executor  # Import agent_executor from main.py

# Initialize LangSmith client
client = Client()

# Initialize the language model
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

# Constants
DATASET_NAME = "Agent Evaluation Dataset V2"
GRADE_PROMPT_DOC_RELEVANCE = (
    "You are a grader. Given the question and retrieved documents, "
    "your task is to score the relevance of the documents to the question.\n\n"
    "Question: {question}\n\nDocuments:\n{documents}\n\n"
    "Give a relevance score from 0 to 1, where:\n"
    "- 1 means the documents are highly relevant to answering the question\n"
    "- 0 means the documents are not relevant\n\nReturn only the score."
)

# Function to retrieve or create a dataset by name
def get_dataset_by_name(dataset_name):
    """
    Retrieves or creates a dataset in LangSmith based on the provided name.

    Args:
        dataset_name (str): Name of the dataset to retrieve or create.

    Returns:
        Dataset: The retrieved or newly created dataset.
    """
    datasets = client.list_datasets()
    for data in datasets:
        if data.name == dataset_name:
            return data
    print(f"Dataset '{dataset_name}' not found. Creating new dataset.")
    return client.create_dataset(dataset_name, description="Dataset for agent evaluation.")

# Retrieve or create the dataset
dataset = get_dataset_by_name(DATASET_NAME)

# Function to predict agent responses based on question input
def agent_predict(inputs):
    """
    Predicts the agent's response for a given question.

    Args:
        inputs (dict): A dictionary containing the question under the key 'question'.

    Returns:
        dict: A dictionary with the predicted answer.
    """
    question_text = inputs["question"]
    response = agent_executor.invoke({"input": question_text})
    return {"answer": response.get("output", "")}

# Sequential evaluation function
def sequential_evaluation(
    agent_predictor, dataset_obj, eval_functions, exp_prefix, eval_description, delay=2
):
    """
    Sequentially evaluates each example in a dataset using specified evaluators.

    Args:
        agent_predictor (callable): Function to predict the agent's response.
        dataset_obj (Dataset): Dataset to evaluate.
        eval_functions (list): List of evaluator functions.
        exp_prefix (str): Prefix for the experiment.
        eval_description (str): Description of the evaluation.
        delay (int): Delay in seconds between evaluations.

    Returns:
        list: Results of the evaluation for each example.
    """
    examples = client.list_examples(dataset_id=dataset_obj.id)
    all_eval_results = []

    for example in examples:
        question_text = example.inputs["question"]
        try:
            eval_result = evaluate(
                agent_predictor,
                data=[example],
                evaluators=eval_functions,
                experiment_prefix=exp_prefix,
                description=eval_description
            )
            all_eval_results.append(eval_result)
            print(f"Evaluation result for question '{question_text}': {eval_result}")
            time.sleep(delay)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error evaluating question '{question_text}': {e}")

    return all_eval_results

def evaluate_token_overlap_correctness(run: Run, example: Example, threshold=0.5) -> dict:
    """
    Evaluates correctness using token overlap between predicted and expected answers.

    Args:
        run (Run): The agent's run containing outputs.
        example (Example): The expected answer example.
        threshold (float): Minimum overlap ratio for correctness.

    Returns:
        dict: Score for token overlap correctness.
    """
    predicted_text = run.outputs.get("answer", "").strip().lower()
    expected_text = example.outputs.get("answer", "").strip().lower()

    tokenizer = re.compile(r'\b\w+\b')
    predicted_tokens = tokenizer.findall(predicted_text)
    expected_tokens = tokenizer.findall(expected_text)

    predicted_counts = Counter(predicted_tokens)
    expected_counts = Counter(expected_tokens)
    shared_tokens = sum((predicted_counts & expected_counts).values())
    total_expected_tokens = sum(expected_counts.values())

    overlap_ratio = shared_tokens / total_expected_tokens if total_expected_tokens > 0 else 0
    score = int(overlap_ratio >= threshold)

    print(f"Predicted: {predicted_text}")
    print(f"Expected: {expected_text}")
    print(f"Overlap ratio: {overlap_ratio:.2f} (Score: {score})")

    return {"score": score, "key": "correctness_token_overlap"}

# Run the evaluation
results = sequential_evaluation(
    agent_predictor=agent_predict,
    dataset_obj=dataset,
    eval_functions=[evaluate_token_overlap_correctness],
    exp_prefix="Agent Correctness Token Overlap Evaluation",
    eval_description="Evaluation of agent's answer correctness using token overlap.",
    delay=2
)

print("Final Sequential Evaluation Results (Correctness - Token Overlap):", results)
