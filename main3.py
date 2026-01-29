import sounddevice as sd
import numpy as np
import whisper
import wave
import requests
import subprocess
import sys
import time

# ---------------- CONFIG ---------------- #

SAMPLE_RATE = 16000
DURATION = 8

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

PIPER_MODEL = "voices/en_US-lessac-medium.onnx"

WAKE_PHRASE = "jamaica"
EXIT_PHRASE = "four times tomorrow"

SYSTEM_PROMPT = """
You are a calm, friendly, NYC tour guide
All responses must pertain to NYC or a related topic.
You speak clearly and naturally.
You give concise but helpful answers.
You do not mention being an AI or language model.
"""

# ---------------------------------------- #


def record_audio():
    print("🎤 Listening...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32
    )
    sd.wait()

    audio_int16 = np.int16(audio * 32767)

    with wave.open("input.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


def speech_to_text():
    model = whisper.load_model("base")
    result = model.transcribe("input.wav")
    return result["text"].strip()


def ask_ollama(user_prompt):
    full_prompt = f"{SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:"

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def speak(text):
    """
    Piper TTS → aplay
    Re-initialized each time for reliability
    """
    piper = subprocess.Popen(
        [
            "piper",
            "--model", PIPER_MODEL,
            "--output-raw"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    aplay = subprocess.Popen(
        [
            "aplay",
            "-r", "22050",
            "-f", "S16_LE",
            "-t", "raw"
        ],
        stdin=piper.stdout
    )

    try:
        piper.stdin.write(text.encode("utf-8"))
        piper.stdin.close()
        aplay.wait()
    except BrokenPipeError:
        pass


def should_ignore(text):
    return WAKE_PHRASE not in text.lower()


def should_exit(text):
    return EXIT_PHRASE in text.lower()


def strip_wake_phrase(text):
    return text.lower().replace(WAKE_PHRASE, "").strip()


# ---------------- MAIN LOOP ---------------- #

if __name__ == "__main__":
    print("🤖 Voice assistant ready.")
    print(f"Say '{WAKE_PHRASE}' to talk.")
    print(f"Say '{EXIT_PHRASE}' to quit.\n")

    speak("Hello. I'm ready when you are.")

    while True:
        try:
            record_audio()
            user_text = speech_to_text()

            if not user_text:
                continue

            print("🧠 You:", user_text)

            if should_exit(user_text):
                speak("Goodbye.")
                break

            if should_ignore(user_text):
                continue

            clean_text = strip_wake_phrase(user_text)

            if not clean_text:
                continue

            reply = ask_ollama(clean_text)
            print("🤖 AI:", reply)

            speak(reply)

            time.sleep(0.2)

        except KeyboardInterrupt:
            print("\nExiting.")
            speak("Goodbye.")
            sys.exit(0)

        except Exception as e:
            print("⚠️ Error:", e)
            time.sleep(1)

