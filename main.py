import sounddevice as sd
import numpy as np
import whisper
# import scipy.io.wavfile as wav
import wave
import requests
import pyttsx3

SAMPLE_RATE = 16000
DURATION = 5
OLLAMA_URL = "http://localhost:11434/api/generate"
engine = pyttsx3.init()

def record_audio():
    # print("🎤 Listening...")
    # audio = sd.rec(
    #     int(DURATION * SAMPLE_RATE),
    #     samplerate=SAMPLE_RATE,
    #     channels=1,
    #     dtype=np.float32
    # )
    # sd.wait()
    # wav.write("input.wav", SAMPLE_RATE, audio)

    print("Listening...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32
    )
    sd.wait()

    # Convert float32 → int16
    audio_int16 = np.int16(audio * 32767)

    with wave.open("input.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    # model = whisper.load_model("base")
    # result = model.transcribe("input.wav")

    print("You said:", result["text"])

def speech_to_text():
    model = whisper.load_model("base")
    result = model.transcribe("input.wav")
    return result["text"]

def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def speak(text):
    
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    record_audio()
    user_text = speech_to_text()
    print("🧠 You:", user_text)

    reply = ask_ollama(user_text)
    print("🤖 AI:", reply)

    speak(reply)
