import sounddevice as sd
import numpy as np
import whisper
import wave
import requests
import subprocess
import sys
import time
import os

# ---------------- CONFIG ---------------- #

SAMPLE_RATE = 16000

WAKE_DURATION = 1.2     # short clip for wake detection
QUERY_DURATION = 3.0    # longer clip for conversation

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

PIPER_MODEL = "voices/en_US-lessac-medium.onnx"

WAKE_PHRASE = "jamaica"
EXIT_PHRASE = "four times tomorrow"

ACTIVE_TIMEOUT = 15  # seconds before going passive again

SYSTEM_PROMPT = """
You are a calm, friendly NYC tour guide.
All responses must relate to New York City or nearby areas.
Speak naturally and clearly.
Be concise but helpful.
Do not mention being an AI or language model.
"""

# ---------------------------------------- #

whisper_model = whisper.load_model("base")


def normalize_numbers(text):
    replacements = {
        "four": "4",
        "for": "4",     # Whisper loves this one
        "to": "2",
        "too": "2",
        "two": "2",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def record_audio(duration):
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
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
    result = whisper_model.transcribe(
        "input.wav",
        language="en",
        fp16=False,
        temperature=0.0,
        condition_on_previous_text=False
    )

    # result = whisper_model.transcribe(
    #     "input.wav",
    #     language="en",
    #     fp16=False,
    #     condition_on_previous_text=False
    # )
    return result["text"].strip().lower()


def ask_ollama(user_prompt):
    full_prompt = f"{SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:"

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "num_predict": 120,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def speak(text):
    piper = subprocess.Popen(
        ["piper", "--model", PIPER_MODEL, "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    aplay = subprocess.Popen(
        ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw"],
        stdin=piper.stdout
    )

    try:
        piper.stdin.write(text.encode("utf-8"))
        piper.stdin.close()
        aplay.wait()
    except BrokenPipeError:
        pass


def wake_detected(text):
    variants = [
        WAKE_PHRASE,
        "jamaican",
        "jamica",
        "jam a car",
        "jamaka",
    ]
    return any(v in text for v in variants)


def should_exit(text):
    text = normalize_numbers(text.lower())
    return "4 times tomorrow" in text
    # return EXIT_PHRASE in text


# ---------------- MAIN LOOP ---------------- #

if __name__ == "__main__":
    print("🤖 Voice assistant ready.")
    print(f"Say '{WAKE_PHRASE}' to talk.")
    print(f"Say '{EXIT_PHRASE}' to quit.\n")

    speak("Hello. I'm ready when you are.")

    mode = "passive"
    last_active = 0

    try:
        while True:
            if mode == "passive":
                print("🎤 Listening for wake word...")
                record_audio(WAKE_DURATION)
            else:
                print("🎤 Listening...")
                record_audio(QUERY_DURATION)

            text = speech_to_text()

            if not text:
                continue

            print("🧠 You:", text)

            if should_exit(text):
                speak("Goodbye.")
                break

            if mode == "passive":
                if wake_detected(text):
                    mode = "active"
                    last_active = time.time()
                    speak("Yes?")
                continue

            # ACTIVE MODE
            last_active = time.time()

            reply = ask_ollama(text)
            print("🤖 AI:", reply)
            speak(reply)

            if time.time() - last_active > ACTIVE_TIMEOUT:
                mode = "passive"

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nExiting.")
        speak("Goodbye.")
        sys.exit(0)
