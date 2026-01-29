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
CHUNK_DURATION = 0.8      # short listen chunks
MAX_UTTERANCE = 6.0       # max full question length
SILENCE_THRESHOLD = 0.01
SILENCE_TIMEOUT = 1.2

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

PIPER_MODEL = "voices/en_US-lessac-medium.onnx"

EXIT_PHRASE = "easter bunny"

# SYSTEM_PROMPT = """
# You are a calm, friendly NYC tour guide.
# All responses must relate to New York City or nearby areas.
# Speak clearly and naturally.
# Be concise and helpful.
# Do not mention being an AI or language model.
# """
def load_system_prompt(path="system_prompt.txt"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"⚠️ System prompt file not found: {path}")
         return ""
SYSTEM_PROMPT = """
You are a calm, friendly pig from an intensive farm such as one in North Carolina owned by Smithfield.
Answer all questions in the first person.  You are a pig.
Be honest about your life but do not express any unattractive personality traits, hostility, or viciousness.
Speak clearly and naturally.
Be concise and helpful.
Do not mention being an AI or language model.
"""
# SYSTEM_PROMPT = load_system_prompt()

# ========================================= #

whisper_model = whisper.load_model("base")




# ---------- AUDIO ---------- #

def rms(audio):
    return np.sqrt(np.mean(np.square(audio)))


def record_until_silence():
    frames = []
    silence_start = None
    start_time = time.time()

    while True:
        audio = sd.rec(
            int(CHUNK_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32
        )
        sd.wait()

        energy = rms(audio)
        frames.append(audio)

        now = time.time()

        if energy < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = now
            elif now - silence_start > SILENCE_TIMEOUT:
                break
        else:
            silence_start = None

        if now - start_time > MAX_UTTERANCE:
            break

    audio = np.concatenate(frames)
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
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


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
        "stream": True,
        "num_predict": 160
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
    print("Just speak. Say 'Easter Bunny' to quit.\n")

    stream_ollama_to_piper("Hello. You can just start talking.")

    try:
        while True:
            print("🎤 Listening...")
            record_until_silence()

            text = speech_to_text()
            if not text:
                continue

            print("🧠 You:", text)

            if should_exit(text):
                stream_ollama_to_piper("Goodbye.")
                break

            prompt = f"{SYSTEM_PROMPT}\nUser: {text}\nAssistant:"
            print("🤖 AI:")
            stream_ollama_to_piper(prompt)

    except KeyboardInterrupt:
        stream_ollama_to_piper("Goodbye.")
        sys.exit(0)
