import os
from gtts import gTTS

# Optional: ElevenLabs (only used if ELEVEN_API_KEY is set in .env)
try:
    import elevenlabs
    from elevenlabs.client import ElevenLabs
    ELEVEN_AVAILABLE = True
except ImportError:
    ELEVEN_AVAILABLE = False

ELEVENLABS_API_KEY = os.environ.get("ELEVEN_API_KEY")


def text_to_speech_with_gtts(input_text, output_filepath="doctor_response.mp3"):
    """
    Converts text to speech using gTTS and saves it as an mp3 file.
    No OS-level playback here — Streamlit will play the file in-browser
    using st.audio(), which works on any deployment (local or cloud).
    """
    tts = gTTS(text=input_text, lang="en", slow=False)
    tts.save(output_filepath)
    return output_filepath


def text_to_speech_with_elevenlabs(input_text, output_filepath="doctor_response.mp3"):
    """
    Converts text to speech using ElevenLabs (higher quality, needs paid/free API key).
    Falls back to gTTS automatically if ElevenLabs isn't configured.
    """
    if not ELEVEN_AVAILABLE or not ELEVENLABS_API_KEY:
        return text_to_speech_with_gtts(input_text, output_filepath)

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="Charlie",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)
    return output_filepath
