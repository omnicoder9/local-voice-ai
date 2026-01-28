import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_ollama(prompt, model="llama3.2"):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]

if __name__ == "__main__":
    reply = ask_ollama("Explain recursion simply")
    print(reply)
