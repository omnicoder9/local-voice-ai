import sounddevice as sd
import numpy as np
import whisper
import wave

SAMPLE_RATE = 16000
DURATION = 5

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

model = whisper.load_model("base")
result = model.transcribe("input.wav")

print("You said:", result["text"])


# import sounddevice as sd
# import numpy as np
# import whisper
# import scipy.io.wavfile as wav

# SAMPLE_RATE = 16000
# DURATION = 5  # seconds

# print("Listening...")
# audio = sd.rec(
#     int(DURATION * SAMPLE_RATE),
#     samplerate=SAMPLE_RATE,
#     channels=1,
#     dtype=np.float32
# )
# sd.wait()

# wav.write("input.wav", SAMPLE_RATE, audio)
# print("Saved input.wav")

# model = whisper.load_model("base")
# result = model.transcribe("input.wav")

# print("You said:", result["text"])
