# Import Libraries
import os  # Import os for file operations
import time  # Import time to set response streaming time
import streamlit as st  # Import Streamlit for UI development
from streamlit_pdf_viewer import pdf_viewer  # Import PDF viewer for displaying PDFs
from langchain_community.document_loaders import PyMuPDFLoader  # Import PDF loader
from langchain.indexes import VectorstoreIndexCreator  # Import index creator
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import text splitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings  # Import OllamaLLM to interact with the LLM




# Initialize Models
llm = OllamaLLM(model="ushinar:1b")  # Initialize the LLM model
embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")  # Initialize the embedding model

# Configure Streamlit App
st.set_page_config(layout="wide")  # Set the page layout to wide with custom menu item
st.title('PDFChatBot')  # Set the title of the app
st.markdown("**PDFChatBot uses AI. Check for mistakes.**")  # Add a markdown note
 
# Initialize Session State
st.session_state.setdefault('messages', [])  # Initialize 'messages' as an empty list
st.session_state.setdefault('chat_history', [])  # Initialize 'chat_history' as an empty list
st.session_state.setdefault('file_uploader_key', 0)  # Initialize file uploader key
st.session_state.setdefault('upload_disabled', False)  # Initialize 'upload_disabled' as False
st.session_state.setdefault('chat_disabled', False)  # Initialize 'chat_disabled' as False

# Start New Chat Button
if st.button('Start New Chat', type='primary'):  # Create a "Start New Chat" button
    for key in list(st.session_state.keys()):  # Iterate over all keys in session state
        if key != 'file_uploader_key':  # Check if the key is not 'file_uploader_key'
            del st.session_state[key]  # Delete the session state key if the condition is met
    st.session_state.file_uploader_key += 1  # Increment the file uploader key to reset the widget
    st.rerun()  # Rerun the app to reflect the changes

# Sidebar for PDF Upload
with st.sidebar:
    st.title("PDF Viewer")  # Set the title for the sidebar
    uploaded_file = st.file_uploader(label="Upload PDF", 
                                     type=["pdf"], 
                                     key=st.session_state.file_uploader_key,
                                     label_visibility="collapsed", 
                                     disabled=st.session_state.upload_disabled, 
                                     on_change=lambda: st.session_state.update(upload_disabled=True))  # Create a file uploader for PDFs
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()  # Read the uploaded file
        pdf_viewer(input=file_bytes, render_text=True, resolution_boost=6)  # Display the PDF in the viewer
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(file_bytes)  # Write the file bytes to a temporary file
        if 'index' not in st.session_state:
            loaders = [PyMuPDFLoader("temp_uploaded.pdf")]  # Load the PDF file
            index_creator = VectorstoreIndexCreator(embedding=embedding_model, text_splitter=RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300))  # Create an index creator
            st.session_state.index = index_creator.from_loaders(loaders) # Load index in session state
            print("\nChunks created successfully!\n\n")  # Print success message
    else:
        st.session_state.update(upload_disabled=False)  # Reset upload_disabled if no file is uploaded
        temp_file_path = "temp_uploaded.pdf"  # Set the path to the temporary file
        if os.path.exists(temp_file_path): # Check if file exists
            os.remove(temp_file_path)  # Remove the file if it exists
            for key in list(st.session_state.keys()):  # Iterate over all keys in session state
              if key != 'file_uploader_key':  # Check if the key is not 'file_uploader_key'
                del st.session_state[key] # Delete session state values
            st.rerun()  # Rerun the app

# Function to extract the first 100 words from the PDF
def extract_intro(temp_file_path):
    # Load the PDF
    loader = PyMuPDFLoader(temp_file_path)
    document = loader.load()
    # Extract text from each page and concatenate
    text = " ".join(page.page_content for page in document)
    # Extract the first 100 words
    words = text.split()[:100]
    return ' '.join(words)

# Function to Retrieve Context
def retrieve_context(question, top_k):
    retriever = st.session_state.index.vectorstore.as_retriever()  # Retrieve vector store
    relevant_docs = retriever.invoke(question)  # Invoke the question on the retriever
    doc_context = "\n\n".join([f"Page {doc.metadata['page']}:\n{doc.page_content}" for doc in relevant_docs[:top_k]])  # Include page number before document content
    return doc_context  # Return the history and document context

# Function to Ask a Question
def ask_question(input_question):
    prev_context = st.session_state.chat_history  # Get previous conversation context
    doc_context = retrieve_context(input_question, top_k=2)  # Retrieve previous and document context
    pdf_intro = extract_intro("temp_uploaded.pdf") # Store the extracted pdf text
    formatted_input = f"""
**Primary Context for chatbot: 
This is a friendly and helpful AI chatbot named BD MedChat to answer user queries based on specific 
context retrieved from the product documentation. It focuses on responding to the **Current User Input.

**Product Introduction:
{pdf_intro}

**Previous Conversation History (ignore if blank): 
{prev_context}

**Retrieved Context product documentation: 
{doc_context}

**Current User Input: 
{input_question}"""

    print(f"\n\n**INPUT TO LLM**\n {formatted_input}")  # Print the input to the LLM
    placeholder = st.empty()  # Create a placeholder in the Streamlit app
    response = ""  # Initialize response as an empty string
    for chunk in llm.stream(input=formatted_input):  # Stream the response from the LLM
        response += chunk  # Append each chunk to the response
        placeholder.chat_message('assistant').markdown(response)  # Display the response
        time.sleep(0.005)  # Sleep for 50 milliseconds
    print(f"\n\n**RESPONE FROM LLM**\n {response}\n\n")  # Print the response
    st.session_state.update(chat_disabled=False)  # Reset chat_disabled
    st.session_state.chat_history.append({"input_question": input_question, "response": response})  # Append the question and response to chat history
    st.session_state.messages.append({'role': 'assistant', 'content': response})  # Append the response to messages

# Main Chat Logic
if uploaded_file is not None:  # Check if a file is uploaded
    if st.session_state.messages:  # Check if there are messages in the session state
        for message in st.session_state.messages:  # Iterate through each message
            st.chat_message(message['role']).markdown(message['content'])  # Display each message
    query = st.chat_input("Enter your question...", disabled=st.session_state.chat_disabled, on_submit=lambda: st.session_state.update(chat_disabled=True))  # Create an input box for user queries
    if query:  # Check if the user has entered a query
        st.session_state.messages.append({'role': 'user', 'content': query})  # Append the query to messages
        st.chat_message('user').markdown(query)  # Display the user query
        ask_question(query)  # Ask the question
        st.session_state.update(chat_disabled=False)  # Reset chat_disabled
        st.rerun()  # Rerun the app
else:
    st.warning("WARNING! Please upload a PDF to start chatting.")  # Display a warning if no PDF is uploaded
