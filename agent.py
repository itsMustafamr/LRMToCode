import os, subprocess
import datetime
from typing import Dict, Any
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

LOG_PATH = "design.log"

def log_event(event: str):
    with open(LOG_PATH, "a") as f: 
        f.write(f"[{datetime.datetime.now()}] {event}\n")

def generate_scad_from_prompt(prompt: str) -> str:
    system_prompt = """
You are an expert CAD engineer. Given the user's design request, output only valid OpenSCAD code.
Start with a comment summarizing the design.

Example Prompt: "Generate a cube 10x10x10 mm."
Example Output:
// Cube 10x10x10 mm
cube([10,10,10]);
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def generate_3d_geometry(prompt: str, confirm: bool = True) -> Dict[str, Any]:
    log_event(f"Prompt received: {prompt}")
    scad_code = generate_scad_from_prompt(prompt)
    log_event(f"SCAD code generated:\n{scad_code}")

    with open("design.scad", "w") as f:
        f.write(scad_code)
    log_event("SCAD file saved as design.scad")

    if confirm:
        print("\n--- Generated OpenSCAD Code ---")
        print(scad_code)
        choice = input("Proceed to generate STL? (y/n): ").strip().lower()
        if choice not in ("y", "yes"):
            log_event("User canceled before STL generation")
            return {"status": "canceled"}

    stl_path = "design.stl"
    try:
        subprocess.run(["openscad", "design.scad", "-o", stl_path], check=True)
        log_event("STL generated successfully")
        return {"status": "success", "model_file": stl_path}
    except subprocess.CalledProcessError:
        log_event("OpenSCAD STL generation failed")
        return {"status": "error", "error_message": "OpenSCAD CLI failed"}
