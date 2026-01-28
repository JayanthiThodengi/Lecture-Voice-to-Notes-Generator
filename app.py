import streamlit as st
import whisper
from transformers import pipeline
import tempfile
import os

# Load models
def load_models():
    speech_model = whisper.load_model("base")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return speech_model, summarizer

speech_model, summarizer = load_models()

st.title("🎤 Lecture Voice-to-Notes Generator")
st.write("Upload a lecture audio file to generate notes and summary")

# Upload audio
audio_file = st.file_uploader("Upload Lecture Audio", type=["mp3", "wav", "m4a"])

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    st.success("Audio uploaded successfully!")

    if st.button("Generate Notes"):
        with st.spinner("Converting speech to text..."):
            result = speech_model.transcribe(audio_path)
            transcript = result["text"]

        st.subheader("📝 Full Transcript")
        st.write(transcript)

        with st.spinner("Generating summary notes..."):
            summary = summarizer(
                transcript,
                max_length=150,
                min_length=60,
                do_sample=False
            )

        st.subheader("📌 Summary Notes")
        st.write(summary[0]["summary_text"])

        # Quiz Generation (simple)
        st.subheader("❓ Sample Quiz Questions")
        st.write("1. What is the main topic discussed in the lecture?")
        st.write("2. Explain one key concept mentioned.")
        st.write("3. What is the conclusion of the lecture?")

        os.remove(audio_path)
