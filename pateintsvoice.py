import os
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def transcribe_with_groq(audio_filepath, stt_model="whisper-large-v3", groq_api_key=None):
    """
    Transcribes an audio file (mp3/wav) to text using Groq's Whisper model.
    audio_filepath: path to an audio file already saved to disk
                     (in the Streamlit app, this comes from st.audio_input).
    """
    client = Groq(api_key=groq_api_key or GROQ_API_KEY)

    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )

    return transcription.text
