# PDFChatBot

A Streamlit-based chatbot application utilizing the `OllamaLLM` language model. This app is designed to answer user queries based on specific context retrieved from a user guide PDF.

## Note

To use the current model, you need to install `ollama` and pull the embedding and llm models.

## Features

- **Streamlit Integration**: Utilizes Streamlit for an interactive web interface.
- **Context Retrieval**: Retrieves context from a user guide PDF.
- **Session Management**: Maintains chat history and context within user sessions.
- **Model Flexibility**: The language model (LLM) can be changed as per requirements.

## Installation

1. **Clone the Repository**:

2. **Install Dependencies**:
    ```sh
    pip install streamlit langchain_ollama langchain_community pymupdf
    ```

## Usage

1. **Run the Application**:
    ```sh
    streamlit run app.py
    ```

2. **Interacting with the ChatBot**:
    - Open the Streamlit app in your browser.
    - Set the primary context for the chatbot.
    - Start chatting!

## Code Overview

- **`app.py`**: The main application file.
    - **Imports**: Imports necessary modules and initializes the language model.
    - **Configuration**: Configures the Streamlit app layout and title.
    - **Session Management**: Initializes session state variables.
    - **PDF Loading**: Loads the user guide PDF and creates an index.
    - **Context Retrieval**: Retrieves relevant context from the PDF based on user queries.
    - **Chat Input**: Handles user input and generates responses using the LLM.
    - **Response Streaming**: Streams the response from the language model.
    - **Chat Display**: Displays chat messages from the session state.

## Customization

To change the language model (LLM) as per your requirements, update the following line in `app.py`:

```python
llm = OllamaLLM(model="your_desired_model")
