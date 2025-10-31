import streamlit as st
import requests

# --- Page Configuration ---
# Set the title and icon for your app's browser tab
st.set_page_config(
    page_title="Ask Your Docs",
    page_icon="🤖"
)

# --- Page Title ---
st.title("🤖 Ask Your Documents")
st.write("This is a simple web app to chat with your documents. Your FastAPI backend must be running in Docker.")

# --- API Endpoint ---
# The URL of your FastAPI server (running in Docker)
API_URL = "http://127.0.0.1:8000/ask"

# --- User Input ---
# Create a text input box for the user's question
user_question = st.text_input("Enter your question:", placeholder="e.g., What is Project Phoenix?")

# --- Submit Button ---
# Create a button labeled "Get Answer"
if st.button("Get Answer"):
    if user_question:
        # Show a "thinking" spinner while we wait
        with st.spinner("Thinking..."):
            try:
                # --- Send the request to the API ---
                payload = {"text": user_question}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    # --- Display the results ---
                    result = response.json()
                    st.subheader("Answer:")
                    st.write(result['answer'])
                    
                    st.subheader("Sources:")
                    for source in result['sources']:
                        st.write(f"- {source}")
                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the FastAPI server. Is it running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a question.")