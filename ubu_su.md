Got it — let’s reset this **from zero** and write **clean, boring, copy-pasteable setup instructions** for a fresh Linux box.
I’ll target **Ubuntu 22.04+**, keep it minimal, and avoid Arch-isms.

This assumes:

* local **STT (Whisper)**
* simple **offline TTS (pyttsx3 + eSpeak)**
* Python venv
* no Docker, no CUDA (CPU-only)

You can paste this straight into a README.

---

# Local Voice AI – Clean Ubuntu Setup (STT + TTS)

## 0. System Requirements

* Ubuntu **22.04 or newer**
* Python **3.10+**
* Working audio output (speakers or headphones)

---

## 1. System Dependencies

Update system packages:

```bash
sudo apt update
sudo apt upgrade -y
```

Install required system libraries:

```bash
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  ffmpeg \
  espeak-ng \
  portaudio19-dev \
  libsndfile1
```

### Verify critical tools

```bash
python3 --version
ffmpeg -version
espeak-ng "hello world"
```

You should **hear audio** from the last command.

---

## 2. Project Setup

Clone your repo and enter it:

```bash
git clone <your-repo-url> local-voice-ai
cd local-voice-ai
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

---

## 3. Python Dependencies

Install required Python packages:

```bash
pip install \
  openai-whisper \
  torch \
  numpy \
  scipy \
  sounddevice \
  pyttsx3 \
  tqdm
```

> ⚠️ Whisper installs **PyTorch** automatically.
> This is CPU-only by default (works everywhere).

---

## 4. Speech-to-Text (Whisper) Test

Create `stt_test.py`:

```python
import whisper

model = whisper.load_model("base")
print("Whisper loaded OK")
```

Run:

```bash
python stt_test.py
```

Expected output:

```
Whisper loaded OK
```

---

## 5. Text-to-Speech Test

Create `tts_test.py`:

```python
import pyttsx3

engine = pyttsx3.init()
engine.say("Text to speech is working")
engine.runAndWait()
```

Run:

```bash
python tts_test.py
```

You should **hear speech**.

---

## 6. Audio Input Test (Microphone)

Test microphone recording:

```bash
arecord -d 3 test.wav
aplay test.wav
```

If this works, Whisper will work.

---

## 7. Common Problems & Fixes

### ❌ `ModuleNotFoundError: scipy / tqdm / sounddevice`

You forgot to install Python deps or venv isn’t active.

```bash
source venv/bin/activate
pip install scipy tqdm sounddevice
```

---

### ❌ `AttributeError: module 'whisper' has no attribute 'load_model'`

Wrong package installed.

Fix:

```bash
pip uninstall whisper -y
pip install openai-whisper
```

Verify:

```bash
python - <<EOF
import whisper
print(whisper.__file__)
EOF
```

Path should include `openai_whisper`.

---

### ❌ `RuntimeError: eSpeak not installed`

Install system TTS engine:

```bash
sudo apt install espeak-ng
```

---

### ❌ No audio output

Check audio system:

```bash
pactl info
speaker-test
```

---

## 8. Recommended Project Structure

```
local-voice-ai/
├── venv/
├── stt.py
├── tts.py
├── requirements.txt
└── README.md
```

Optional `requirements.txt`:

```
openai-whisper
torch
numpy
scipy
sounddevice
pyttsx3
tqdm
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 9. Notes

* This setup is **fully offline**
* Voice quality is **basic** (eSpeak)
* Whisper runs on **CPU**
* Works identically on Ubuntu, Debian, Pop!_OS

---

## 10. Next Logical Upgrades (Optional)

* Replace pyttsx3 → **Piper TTS** (better voices)
* Add streaming microphone input
* Wire STT → LLM → TTS loop
* GPU acceleration for Whisper

If you want, I can:

* convert this into a **Dockerfile**
* rewrite for **WSL**
* upgrade TTS to **Piper**
* make a **one-command install script**

Just tell me what’s next 👍
