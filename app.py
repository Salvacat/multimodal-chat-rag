"""
This module creates a Gradio-based interface for interacting with an agent
that processes questions and retains a conversation history.
"""

import gradio as gr
from main import agent_executor
from memory import conversation_history

def process_question(question):
    """
    Processes a user's question by invoking the agent and retrieving a response.

    Args:
        question (str): The question input by the user.

    Returns:
        str: The response from the agent, or an error message if an issue occurs.
    """
    response = None
    try:
        response = agent_executor.invoke({"input": question})
    except KeyError as e:
        print(f"Error encountered: {e}. Attempting to clear memory and retry.")
        conversation_history.clear()
    return response["output"] if response else "An error occurred."

def reset_conversation_history():
    """
    Clears the conversation history to reset the interaction state.

    Returns:
        str: Confirmation message indicating the conversation history was reset.
    """
    conversation_history.clear()
    return "Conversation history reset."

# Gradio output component for reset feedback
reset_output = gr.Textbox(label="Reset Status")

# Define the Gradio interface
with gr.Blocks() as interface:
    # Question processing component
    input_textbox = gr.Textbox(label="Ask a question:")
    output_textbox = gr.Textbox(label="Agent Response")
    submit_button = gr.Button("Submit")
    submit_button.click(fn=process_question, inputs=input_textbox, outputs=output_textbox)  # pylint: disable=E1101

    # Reset button and output
    reset_button = gr.Button("Reset Conversation History")
    reset_button.click(fn=reset_conversation_history, outputs=reset_output)  # pylint: disable=E1101

    # Display reset status
    reset_output.render()

# Launch the app
if __name__ == "__main__":
    interface.launch(share=True)
