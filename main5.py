import sounddevice as sd
import numpy as np
import whisper
import wave
import requests
import subprocess
import sys
import time
import json

# ================= CONFIG ================= #

SAMPLE_RATE = 16000

WAKE_DURATION = 1.2
QUERY_DURATION = 4.0

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

PIPER_MODEL = "voices/en_US-lessac-medium.onnx"

WAKE_PHRASE = "Dear Oliver"
EXIT_PHRASE = "The Easter Bunny"

ACTIVE_TIMEOUT = 15

SYSTEM_PROMPT = """
You are a calm, friendly NYC tour guide.
All responses must relate to New York City or nearby areas.
Speak clearly and naturally.
Be concise and helpful.
Do not mention being an AI or language model.
"""

# ========================================= #

whisper_model = whisper.load_model("base")


# ---------- AUDIO ---------- #

def record_audio(duration):
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32
    )
    sd.wait()

    audio = np.int16(audio * 32767)

    with wave.open("input.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def speech_to_text():
    result = whisper_model.transcribe(
        "input.wav",
        language="en",
        fp16=False,
        temperature=0.0,
        condition_on_previous_text=False
    )
    return result["text"].strip().lower()


# ---------- NORMALIZATION ---------- #

def normalize(text):
    replacements = {
        "four": "4",
        "for": "4",
        "to": "2",
        "too": "2",
        "two": "2",
        "times": "times",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def wake_detected(text):
    text = normalize(text)
    return any(
        variant in text
        for variant in [
            WAKE_PHRASE,
            "jamaican",
            "jamica",
            "jam a car",
            "jamaka",
        ]
    )


def should_exit(text):
    return EXIT_PHRASE in normalize(text)


# ---------- STREAMING TTS ---------- #

def stream_ollama_to_piper(prompt):
    piper = subprocess.Popen(
        ["piper", "--model", PIPER_MODEL, "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=0
    )

    aplay = subprocess.Popen(
        ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw"],
        stdin=piper.stdout
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True
    }

    with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode())
            token = data.get("response", "")

            if token:
                try:
                    piper.stdin.write(token.encode("utf-8"))
                    piper.stdin.flush()
                except BrokenPipeError:
                    break

    piper.stdin.close()
    aplay.wait()


# ================= MAIN ================= #

if __name__ == "__main__":
    print("🤖 Voice assistant ready.")
    print(f"Wake phrase: {WAKE_PHRASE}")
    print(f"Exit phrase: {EXIT_PHRASE}\n")

    stream_ollama_to_piper("Hello. I'm ready when you are.")

    mode = "passive"
    last_active = 0

    try:
        while True:
            if mode == "passive":
                record_audio(WAKE_DURATION)
            else:
                record_audio(QUERY_DURATION)

            text = speech_to_text()
            if not text:
                continue

            print("🧠 You:", text)

            if should_exit(text):
                stream_ollama_to_piper("Goodbye.")
                break

            if mode == "passive":
                if wake_detected(text):
                    mode = "active"
                    last_active = time.time()
                    stream_ollama_to_piper("Yes?")
                continue

            # ACTIVE MODE
            last_active = time.time()

            full_prompt = f"{SYSTEM_PROMPT}\nUser: {text}\nAssistant:"
            print("🤖 AI (streaming):")
            stream_ollama_to_piper(full_prompt)

            if time.time() - last_active > ACTIVE_TIMEOUT:
                mode = "passive"

    except KeyboardInterrupt:
        stream_ollama_to_piper("Goodbye.")
        sys.exit(0)
