import streamlit as st
import requests
import json
import sseclient
import time

st.title("Thoughtful AI Chatbot")
st.write("Chat with a Thoughtful AI assistant!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to handle SSE streaming
def stream_response(response):
    message_placeholder = st.empty()
    full_response = ""
    
    # Read the response line by line
    for line in response.iter_lines():
        if line:
            # Decode the line and remove "data: " prefix
            try:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove "data: " prefix
                    full_response += data
                    message_placeholder.markdown(full_response + "▌")
            except Exception as e:
                st.error(f"Error processing stream: {str(e)}")
                break
    
    message_placeholder.markdown(full_response)
    return full_response

# Define the API URL
API_URL = "http://localhost:8000"  # FastAPI server address

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send request to FastAPI backend
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{API_URL}/chat",  # Use the API_URL constant
                params={"query": prompt},
                stream=True
            )
            response.encoding = 'utf-8'
            
            full_response = stream_response(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {str(e)}") 