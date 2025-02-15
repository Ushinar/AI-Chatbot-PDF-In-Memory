# Import Libraries
import streamlit as st  # Import Streamlit for UI development
import time  # Import time to set response streaming time
from langchain_community.document_loaders import PyMuPDFLoader  # Import PDF loader
from langchain.indexes import VectorstoreIndexCreator  # Import index creator
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import text splitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings  # Import OllamaLLM to interact with the LLM




# Initialize Models
llm = OllamaLLM(model="llama3.2:1b")  # Initialize the LLM model
embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")  # Initialize the embedding model


# Configure Streamlit App
st.set_page_config(layout="wide")  # Set the page layout to wide
st.title('PDFChatBot')  # Set the title of the app


# Initialize Session State
if 'messages' not in st.session_state:  # Check if 'messages' key is not in session state
    st.session_state.messages = []  # Initialize 'messages' as an empty list

if 'index' not in st.session_state:  # Check if 'index' key is not in session state
    pdf_path = "./medlight_6.pdf"  # Define the path to the PDF file
    loaders = [PyMuPDFLoader(pdf_path)]  # Load the PDF file
    ## For multiple PDFs:
    ## pdf_paths = ["path/to/your/pdf", "path/to/your/pdf", "path/to/your/pdf"]
    ## loaders = [PyMuPDFLoader(path) for path in pdf_paths]
    index_creator = VectorstoreIndexCreator(  # Create an index creator
        embedding=embedding_model,  # Use the embedding model
        text_splitter=RecursiveCharacterTextSplitter(  # Use the text splitter
            chunk_size=3000,  # Set chunk size to 1000
            chunk_overlap=300))  # Set chunk overlap to 100
    print("\nChunks created successfully!\n\n")  # Print success message
    st.session_state.index = index_creator.from_loaders(loaders)  # Create the index from loaders

if 'chat_history' not in st.session_state:  # Check if 'chat_history' key is not in session state
    st.session_state.chat_history = []  # Initialize 'chat_history' as an empty list


# Function to Retrieve Context
def retrieve_context(question, top_k):
    retriever = st.session_state.index.vectorstore.as_retriever()  # Retrieve vector store
    relevant_docs = retriever.invoke(question)  # Invoke the question on the retriever
    # Get top_k relevant documents
    doc_context = "\n\n".join([
        f"Page {doc.metadata['page']}:\n{doc.page_content}" for doc in relevant_docs[:top_k]])  # Include page number before document content
    return doc_context  # Return the history and document context


# Function to Ask a Question
def ask_question(input_question):
    prev_context = st.session_state.chat_history
    doc_context = retrieve_context(input_question, top_k=2)  # Retrieve previous and document context
    formatted_input = f"""
**Primary Context for chatbot: 
This is a friendly and helpful AI chatbot named BD MedChat to answer user queries based on specific 
context retrieved from the user guide. It focuses on responding to the **Current User Input.
    
**Previous Conversation Context (ignore if blank): 
{prev_context}
    
**Retrieved Context from user guide: 
{doc_context}
    
**Current User Input: 
{input_question}"""  # Format the input string for the LLM
    
    print(f"\n\n**INPUT TO LLM**\n {formatted_input}")  # Print the input to the LLM
    placeholder = st.empty()  # Create a placeholder in the Streamlit app
    response = ""  # Initialize response as an empty string
    for chunk in llm.stream(input=formatted_input):  # Stream the response from the LLM
        response += chunk  # Append each chunk to the response
        placeholder.chat_message('assistant').markdown(response)  # Display the response
        time.sleep(0.05)  # Sleep for 50 milliseconds

    print(f"\n\n**RESPONE FROM LLM**\n {response}\n\n")  # Print the response
    st.session_state.chat_history.append(  # Append the question and response to chat history
        {"input_question": input_question, "response": response})
    st.session_state.messages.append({'role': 'assistant', 'content': response})  # Append the response to messages


# Main Chat Logic
query = st.chat_input("Enter your question...")  # Create an input box for user queries
if query:  # If there is a query
    st.session_state.messages.append({'role': 'user', 'content': query})  # Append the query to messages
    for message in st.session_state.messages:  # Loop through messages
        st.chat_message(message['role']).markdown(message['content'])  # Display each message
    ask_question(query)  # Ask the question
    
