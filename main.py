import streamlit as st
import re
from analysis import extract_video_id, get_answer

# Set page config
st.set_page_config(page_title="YouTube Q&A", layout="wide")

st.title("🤖 AI-Powered YouTube Transcript Analysis")
st.markdown("""
Ask context-aware questions about any YouTube video!  
This app extracts the transcript, processes it, and lets you query it using an LLM.
""")

# Input YouTube URL
youtube_url = st.text_input("Enter YouTube Video URL")

video_id = extract_video_id(youtube_url)
print("video_id:", video_id)

# Sidebar for question input
st.sidebar.header("Ask a Question")
user_question = st.sidebar.text_input("Enter your question:")
ask_button = st.sidebar.button("Ask")

# Display YouTube video
if video_id:
    st.video(f"https://www.youtube.com/embed/{video_id}")

# Process the transcript and answer the question
if video_id and ask_button and user_question:
    try:
        result = get_answer(video_id = video_id, language="en", search_type="similarity", k=4, query=user_question)

        # Display result
        st.subheader("Answer:")
        st.write(result)

    except Exception as e:
        st.error(f"Error processing video: {e}")
