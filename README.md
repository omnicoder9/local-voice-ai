# local-voice-ai

python main2.py
---


python -m venv venv
source venv/bin/activate

pip install \
  sounddevice \
  numpy \
  pyttsx3 \
  requests

pip install openai-whisper


python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
EOF

pip install openai-whisper --no-deps
python - <<EOF
import whisper
model = whisper.load_model("base")
print("Whisper loaded OK")
EOF

pip install tqdm numpy regex ffmpeg-python

====

python -m venv venv
source venv/bin/activate
python --version

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper

---
python - <<EOF
import whisper
model = whisper.load_model("base")
print("Whisper loaded OK")
EOF

---

pip install \
  sounddevice \
  numpy \
  pyttsx3 \
  requests

pip install piper-tts

curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
