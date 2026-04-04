import streamlit as st
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import backend logic
# Import backend logic
from medical_assistant import get_response, COLLECTION_NAME, listen_and_transcribe, speak_response

# Page Config
st.set_page_config(
    page_title="Medical Assistant",
    page_icon="�",
    layout="wide"
)

# Sidebar - Language Selection
st.sidebar.title("⚙️ Settings")
language_map = {
    "English": "en",
    "Français": "fr",
    "العربية": "ar",
    "Tounsi": "tn"
}
selected_lang_name = st.sidebar.selectbox(
    "Choose Language / Choisir la langue / اختر اللغة",
    list(language_map.keys())
)
selected_lang_code = language_map[selected_lang_name]

# Voice Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🎤 Voice Settings")
voice_response_enabled = st.sidebar.checkbox("Enable Voice Response", value=True)

# Main Interface
st.title("� Medical Assistant: Fainting and Syncope Specialist")
st.markdown("""
Welcome! I am your **Medical Assistant**.
I provide immediate instructions for fainting (syncope) and dizziness. **I am not a doctor**, but I can guide you through first aid steps.
""")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
prompt = st.chat_input("I feel dizzy / Someone just fainted...")

# Voice Input Button
if st.sidebar.button("🎤 Speak / Parler / تحدث"):
    with st.spinner("Listening... Speak now!"):
        voice_text = listen_and_transcribe()
        if voice_text:
            prompt = voice_text
        else:
            st.warning("No speech detected or microphone error.")

if prompt:
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Format history for backend: list of tuples (user, bot)
        # Filter for completed user-assistant pairs
        backend_history = []
        temp_msgs = st.session_state.messages[:-1] # Exclude current prompt
        for i in range(0, len(temp_msgs) - 1, 2):
            if temp_msgs[i]["role"] == "user" and temp_msgs[i+1]["role"] == "assistant":
                backend_history.append((temp_msgs[i]["content"], temp_msgs[i+1]["content"]))

        with st.spinner("Thinking..."):
            try:
                # Call backend with selected language
                response = get_response(prompt, backend_history, language=selected_lang_code)
                
                # Simulate stream effect
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                st.error(f"Error: {e}")
                full_response = "Sorry, I encountered an error."

    # Add assistant message to state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Text-to-Speech Output
    if voice_response_enabled:
        # Check if the last interaction was voice-triggered, OR just always speak if enabled
        # For simplicity, if the toggle is ON, we speak the response.
        speak_response(full_response)

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created by **Jihed Bakalti**")
st.sidebar.code("v2.0")
