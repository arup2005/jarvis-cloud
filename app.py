import os
import tempfile
from flask import Flask, request, jsonify, send_file
from groq import Groq
from serpapi.google_search import GoogleSearch
from elevenlabs.client import ElevenLabs
from mtranslate import translate

app = Flask(__name__)

# 🔑 ENV VARIABLES
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
tts_client = ElevenLabs(api_key=ELEVEN_API_KEY)

memory = {}

def translate_text(text):
    try:
        return translate(text, "en")
    except:
        return text

def search_google(query):
    try:
        params = {"q": query, "api_key": SERPAPI_KEY}
        results = GoogleSearch(params).get_dict()

        if "answer_box" in results:
            return results["answer_box"].get("answer") or results["answer_box"].get("snippet")

        if "organic_results" in results:
            return results["organic_results"][0].get("snippet")

    except Exception as e:
        return str(e)

    return None

def ask_ai(user_id, prompt):
    if user_id not in memory:
        memory[user_id] = []

    memory[user_id].append({"role": "user", "content": prompt})
    memory[user_id] = memory[user_id][-5:]

    response = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[{"role": "system", "content": "You are Jarvis. Be short and smart."}] + memory[user_id]
    )

    reply = response.choices[0].message.content
    memory[user_id].append({"role": "assistant", "content": reply})

    return reply

def jarvis_brain(user_id, text):
    result = search_google(text)
    if result:
        return result
    return ask_ai(user_id, text)

def generate_audio(text):
    audio_stream = tts_client.text_to_speech.convert(
        voice_id="CwhRBWXzGAHq8TQ4Fs17",
        model_id="eleven_multilingual_v2",
        text=text
    )

    audio_bytes = b"".join(audio_stream)

    file_path = "/tmp/output.mp3"
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    return file_path

# ✅ TEXT API
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_id = data.get("user_id", "esp32")
    text = translate_text(data.get("text", ""))

    reply = jarvis_brain(user_id, text)

    return jsonify({"reply": reply})

# ✅ VOICE API
@app.route("/voice", methods=["POST"])
def voice():
    data = request.json
    user_id = data.get("user_id", "esp32")
    text = translate_text(data.get("text", ""))

    reply = jarvis_brain(user_id, text)
    audio_path = generate_audio(reply)

    return send_file(audio_path, mimetype="audio/mpeg")

@app.route("/")
def home():
    return "Jarvis Cloud Running 🚀"