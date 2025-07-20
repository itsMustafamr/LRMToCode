#!/usr/bin/env python3
"""
Async CAD-THOUGHTS Generator
Reads a list of design prompts and uses DeepSeek-R1 to auto-generate
Q-A-CoT pairs in JSON format: {prompt, code_scad, chain_of_thought}.
"""
import argparse
import asyncio
import aiohttp
import pandas as pd
import random
import time
import json
from collections import deque
from tqdm import tqdm
from google import genai
from google.genai import types

# Rate-limit settings
MAX_CONCURRENT_REQUESTS = 3
REQUEST_TIMESTAMPS = deque()

# API endpoints & keys
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_API_KEY = json.load(open("keys.json"))['together']

async def enforce_rate_limit():
    while True:
        now = time.time()
        # drop timestamps older than 1s
        while REQUEST_TIMESTAMPS and now - REQUEST_TIMESTAMPS[0] > 1:
            REQUEST_TIMESTAMPS.popleft()
        if len(REQUEST_TIMESTAMPS) < MAX_CONCURRENT_REQUESTS:
            REQUEST_TIMESTAMPS.append(now)
            return
        await asyncio.sleep(0.05)

async def fetch_completion(session, sem, prompt, system_prompt, model_name, temperature=0.6, top_p=0.95, max_tokens=12000):
    """
    Call Together API for DeepSeek-R1 to generate OpenSCAD + CoT JSON.
    """
    async with sem:
        await enforce_rate_limit()
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_completion_tokens": max_tokens
        }
        async with session.post(TOGETHER_API_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"prompt": prompt, "error": f"HTTP {resp.status}: {text}"}
            data = await resp.json()
            # Together returns choices list
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"prompt": prompt, "response": content}

async def generate_cad_thoughts(prompts, args):
    # fixed model/provider for CAD-THOUGHTS
    model_name = "deepseek-ai/DeepSeek-R1"
    system_prompt = (
        "You are an expert CAD engineer. "
        "For the following design specification, generate an OpenSCAD module and then provide a detailed step-by-step chain-of-thought explaining your design decisions. "
        "Output **only** valid JSON** with keys 'code_scad' (string) and 'chain_of_thought' (array of strings)."
    )
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_completion(session, sem, p, system_prompt, model_name,
                                  args.temperature, args.top_p, args.max_tokens)
                 for p in prompts]
        pbar = tqdm(total=len(tasks), desc="Generating CAD-THOUGHTS")
        cad_data = []
        with open(args.partial_file, "a", encoding="utf-8") as partial:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                partial.write(json.dumps(result, ensure_ascii=False) + "\n")
                # parse JSON response
                if 'response' in result:
                    try:
                        parsed = json.loads(result['response'])
                        cad_data.append({
                            'prompt': result['prompt'],
                            'code_scad': parsed.get('code_scad'),
                            'chain_of_thought': parsed.get('chain_of_thought')
                        })
                    except Exception:
                        cad_data.append({
                            'prompt': result['prompt'],
                            'error': 'Failed to parse JSON',
                            'raw_response': result['response']
                        })
                else:
                    cad_data.append(result)
                pbar.update(1)
        return cad_data

async def main():
    parser = argparse.ArgumentParser(description="Generate CAD-THOUGHTS with DeepSeek-R1")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to text file with one design prompt per line")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save CAD-THOUGHTS JSON list")
    parser.add_argument("--partial_file", type=str, default="cad_partial.jsonl",
                        help="File to append intermediate results")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=12000)
    args = parser.parse_args()

    # load prompts
    with open(args.input_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    # generate
    cad_thoughts = await generate_cad_thoughts(prompts, args)

    # save full output
    with open(args.output_file, 'w', encoding='utf-8') as out:
        json.dump(cad_thoughts, out, ensure_ascii=False, indent=2)
    print(f"Done! CAD-THOUGHTS saved to {args.output_file}")

if __name__ == "__main__":
    asyncio.run(main())
