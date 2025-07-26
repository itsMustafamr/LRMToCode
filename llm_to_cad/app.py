from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from openscad_runner import RenderMode, OpenScadRunner

# -------------------------------------------------
# Config & OpenAI client
# -------------------------------------------------
load_dotenv()

with open("keys.json", "r") as f:
    keys = json.load(f)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", keys.get("gpt")))

SYSTEM_PROMPT = (
    "You are an expert CAD engineer who writes clear, idiomatic OpenSCAD. "
    "Respond ONLY with valid OpenSCAD code – no markdown fences, no comments."
)

# -------------------------------------------------
# Core generation helper
# -------------------------------------------------

def generate_scad(request_str: str) -> str:
    """Ask the LLM for plain OpenSCAD code (no comments)."""
    user_msg = (
        f"Create the OpenSCAD code to generate the 3D model for a {request_str}. "
        "Answer ONLY with the code."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content.strip()

# -------------------------------------------------
# Flask setup
# -------------------------------------------------
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
        scad_code = generate_scad(text)

        # remove markdown/code fences if model sneaks them in
        if "```" in scad_code:
            scad_code = scad_code.split("```")[1]
        elif "`" in scad_code:
            scad_code = scad_code.split("`")[1]
        scad_code = re.sub(r"openscad", "", scad_code, flags=re.IGNORECASE)

        # persist to file
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs("scad_scripts", exist_ok=True)
        os.makedirs(os.path.join("static", "images"), exist_ok=True)
        scad_path = os.path.join("scad_scripts", f"{timestamp}.scad")
        with open(scad_path, "w", encoding="utf-8") as f:
            f.write(scad_code)

        # render preview PNG
        img_path = os.path.join("static", "images", f"{timestamp}.png")
        osr = OpenScadRunner(scad_path, img_path, render_mode=RenderMode.preview, imgsize=(800, 600))
        osr.run()

        if osr.good():
            return jsonify({
                "image": f"{timestamp}.png",
                "filename": timestamp,
                "code": scad_code,
                "source": "OpenAI GPT-4o"
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