from flask import Flask, request, jsonify, send_file
from groq import Groq
from elevenlabs.client import ElevenLabs
from serpapi import GoogleSearch
from mtranslate import translate
import requests
import uuid, os, json

app = Flask(__name__)

# 🔑 ENV KEYS
GROQ_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
SERP_KEY = os.getenv("SERPAPI_KEY")

groq = Groq(api_key=GROQ_KEY)
tts = ElevenLabs(api_key=ELEVEN_KEY)

MEMORY_FILE = "memory.json"

# 🧠 MEMORY
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return {"history": []}

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f)

# 🌍 TRANSLATE
def to_english(text):
    try:
        return translate(text, "en")
    except:
        return text

# 🔎 GOOGLE SEARCH
def search_google(query):
    params = {
        "q": query,
        "api_key": SERP_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    try:
        return results["organic_results"][0]["snippet"]
    except:
        return ""

# ⚡ FAST SPEECH (Groq Whisper API)
def speech_to_text(file_path):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    headers = {
        "Authorization": f"Bearer {GROQ_KEY}"
    }

    files = {
        "file": open(file_path, "rb")
    }

    data = {
        "model": "whisper-large-v3"
    }

    response = requests.post(url, headers=headers, files=files, data=data)
    return response.json()["text"]

# 🎤 SPEECH ROUTE
@app.route("/speech", methods=["POST"])
def speech():
    file = request.files['audio']
    fname = f"{uuid.uuid4()}.wav"
    file.save(fname)

    text = speech_to_text(fname)
    os.remove(fname)

    translated = to_english(text)

    return jsonify({
        "original": text,
        "translated": translated
    })

# 🤖 MAIN AI
@app.route("/ask", methods=["POST"])
def ask():
    user_text = request.json["text"]

    # 🔎 search trigger
    use_search = any(w in user_text.lower() for w in [
        "what", "who", "when", "where", "price", "news", "today"
    ])

    search_data = search_google(user_text) if use_search else ""

    memory = load_memory()
    history = memory["history"][-5:]

    system_prompt = f"""
You are Jarvis AI.

Use this info if available:
{search_data}

Respond naturally in English.

Return JSON:
{{"reply":"...", "emotion":"happy/sad/angry/neutral/excited"}}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": user_text})

    res = groq.chat.completions.create(
        messages=messages,
        model="moonshotai/kimi-k2-instruct-0905"
    )

    content = res.choices[0].message.content

    try:
        data = json.loads(content)
    except:
        data = {"reply": content, "emotion": "neutral"}

    reply = data["reply"]
    emotion = data["emotion"]

    # 💾 SAVE MEMORY
    memory["history"].append({"role": "user", "content": user_text})
    memory["history"].append({"role": "assistant", "content": reply})
    save_memory(memory)

    # 🔊 TTS
    audio = tts.generate(text=reply, voice="Rachel")

    fname = f"{uuid.uuid4()}.mp3"
    with open(fname, "wb") as f:
        f.write(audio)

    return jsonify({
        "reply": reply,
        "emotion": emotion,
        "audio": "/audio/" + fname
    })

@app.route("/audio/<file>")
def audio(file):
    return send_file(file, mimetype="audio/mpeg")

# HEALTH CHECK (for UptimeRobot)
@app.route("/")
def home():
    return "Jarvis Running 🚀"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
