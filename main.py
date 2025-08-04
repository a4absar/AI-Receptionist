#AI Receptionist 
# Take down messages, maybe check calendar invites, provide information
# Schedule a meeting/appointment, google calendar, ask for information, is the doctor available

# Used to record audio from the microphone (sounddevice) and save/read audio files (soundfile)
import sounddevice as sd 
import soundfile as sf

# SDK to use OpenAI APIs (e.g., Whisper for transcription, GPT for responses)
import openai

# Used to detect keyboard events, e.g., waiting for spacebar press to start recording.
import keyboard

# Creates temporary files for intermediate storage (like saving audio recordings before transcription).
import tempfile

# For handling environment variables and file paths.
import os

# ElevenLabs TTS SDK to generate and play audio from text
from elevenlabs import generate, play, set_api_key

# call_agent() – initializes a Zapier-integrated LangChain agent.
# answer_the_call() – uses prompt templating to reply in a receptionist style.
from execute_ai import call_agent, answer_the_call

duration = 10  # duration of each recording in seconds
fs = 44100  # sample rate
channels = 1  # number of channels
sd.default.samplerate = fs
sd.default.channels = 2 # sets device to stereo input (but function records mono)

os.environ["OPENAI_API_KEY"] = ""  # Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY") # Configure OpenAI SDK with key

set_api_key("") # Set your ElevenLabs API key here
os.environ["ZAPIER_NLA_API_KEY"] = os.environ.get("ZAPIER_NLA_API_KEY", "") #Optional: sets the Zapier NLA key if not already set.

# Starts recording from the microphone.
# Records for duration seconds using the specified sample rate and channels.
def record_audio(duration, fs, channels):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait() # Waits until recording is finished
    print("Finished recording.")
    return recording

# Saves recorded audio to a temp file, transcribes it using OpenAI Whisper (whisper-1), and returns plain text.
def transcribe_audio(recording, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, recording, fs)  # Save recording to temp WAV file
        temp_audio.close()
        with open(temp_audio.name, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file) # Transcribe using Whisper API
        os.remove(temp_audio.name) # Clean up temp file
    return transcript["text"].strip()

# Converts text to speech using ElevenLabs and plays it back using their SDK.
# You can change voice to others like "Rachel", "Adam", etc.
def play_generated_audio(text, voice="Bella", model="eleven_monolingual_v1"):
    audio = generate(text=text, voice=voice, model=model)
    play(audio)

# Ensures the script runs only when executed directly (not when imported).
if __name__ == '__main__':

    # Plays a friendly welcome message using ElevenLabs.
    inital_text = "Hi! You have reached Dunder Mifflin's office, this is Pam AI. What can I do for you today?"
    play_generated_audio(inital_text)

    while True:
        print("Press spacebar to start recording.")
        keyboard.wait("space")   # waits for the user to press the spacebar
        recorded_audio = record_audio(duration, fs, channels) # Record user's voice
        message = transcribe_audio(recorded_audio, fs)  # Convert audio to text
        print(f"You: {message}") # Show user message
        wait_text = "Please hold on a minute, thank you!"
        play_generated_audio(wait_text) # Play polite hold message
        agent = call_agent()  # Create Zapier agent
        task_info = agent.run(message) # Run agent on user's transcribed message
        answer_chain = answer_the_call() # Create friendly LLM chain
        answer = answer_chain.predict(INFO = task_info) # Generate receptionist-style response
        play_generated_audio(answer) # Speak the final response to the caller