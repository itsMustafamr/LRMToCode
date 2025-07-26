from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from openscad_runner import RenderMode, OpenScadRunner
import requests

# -------------------------------------------------
# Config & OpenAI client
# -------------------------------------------------
load_dotenv()

def _load_keys(json_path: str = "keys.json") -> dict:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            keys_data = json.load(f)
    except FileNotFoundError:
        keys_data = {}

    provider_to_env = {
        "gpt": "OPENAI_API_KEY",
        "claude": "CLAUDE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "together": "TOGETHER_API_KEY",
    }

    for provider_key, env_var in provider_to_env.items():
        if provider_key in keys_data and not os.getenv(env_var):
            os.environ[env_var] = keys_data[provider_key]

    if not os.getenv("DEEPSEEK_API_KEY") and os.getenv("TOGETHER_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = os.getenv("TOGETHER_API_KEY")

    return keys_data

keys = _load_keys()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", keys.get("gpt")))

SYSTEM_PROMPT = (
    "You are an expert CAD engineer who writes clear, idiomatic OpenSCAD. "
    "Respond ONLY with valid OpenSCAD code – no markdown fences, no comments."
)

def generate_scad(request_str: str, provider: str = "gpt") -> tuple[str, str]:
    provider = (provider or "gpt").lower()
    user_msg = (
        f"Create the OpenSCAD code to generate the 3D model for a {request_str}. "
        "Answer ONLY with the code."
    )

    if provider in {"gpt", "openai"}:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content.strip(), "OpenAI GPT-4o"

    if provider == "claude":
        api_key = os.getenv("CLAUDE_API_KEY", keys.get("claude"))
        if not api_key:
            raise ValueError("CLAUDE_API_KEY not configured")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2048,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": user_msg}]}
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        parts = data.get("content", [])
        scad = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        return scad.strip(), "Anthropic Claude-3.5"

    if provider in {"together", "deepseek"}:
        api_key = os.getenv("TOGETHER_API_KEY", keys.get("together"))
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not configured")

        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip(), "Together / DeepSeek V3"

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", keys.get("gemini"))
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-1.5-pro:generateContent?key=" + api_key
        )
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {"role": "user", "parts": [{"text": user_msg}]}
            ]
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        candidate = (resp.json().get("candidates") or [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        scad = "".join(p.get("text", "") for p in parts)
        return scad.strip(), "Google Gemini 1.5 Pro"

    raise ValueError(f"Unsupported provider: {provider}")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty prompt"})

    try:
        provider = request.form.get("provider", "gpt").lower()
        scad_code, source = generate_scad(text, provider)

        # strip code fences robustly
        m = re.search(r"```(?:scad|openscad)?\s*([\s\S]*?)```", scad_code, flags=re.IGNORECASE)
        if m:
            scad_code = m.group(1).strip()
        scad_code = scad_code.strip("`")
        scad_code = re.sub(r"\bopenscad\b", "", scad_code, flags=re.IGNORECASE).strip()

        # persist to file
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs("scad_scripts", exist_ok=True)
        os.makedirs(os.path.join("static", "images"), exist_ok=True)
        scad_path = os.path.join("scad_scripts", f"{ts}.scad")
        with open(scad_path, "w", encoding="utf-8") as f:
            f.write(scad_code)

        # render preview
        img_path = os.path.join("static", "images", f"{ts}.png")
        osr = OpenScadRunner(scad_path, img_path, render_mode=RenderMode.preview, imgsize=(800, 600))
        osr.run()

        if osr.good():
            return jsonify({
                "image": f"{ts}.png",
                "filename": ts,
                "code": scad_code,
                "source": source
            })
        else:
            return jsonify({"error": "OpenSCAD rendering failed", "code": scad_code})

    except Exception as exc:
        return jsonify({"error": f"Server error: {exc}"})

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory("scad_scripts", f"{filename}.scad", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=False)
