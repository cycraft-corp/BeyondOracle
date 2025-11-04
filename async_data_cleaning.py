import asyncio
import json
import math
import re
import textwrap
import traceback
from typing import List, Dict, Any
from openai import AsyncOpenAI

# ——— LLM client setup ———
MODEL = "gpt-4o-mini"
OPENAI_API_KEY = ""

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
)

PATH_TO_UNIT_TEST = "./resources/unit_test_generation_results.jsonl"
PATH_TO_OUTPUT = "./resources/cleaning_results.jsonl"

THREAD_NUM = 4

# ——— Helpers ———
def load_checking_function_from_string(func_code: str, func_name: str = "verifier_function"):
    local_env: Dict[str, Any] = {}
    global_env = {
        "__builtins__": __builtins__,
        "re": re,
        "json": json,
        "math": math,
        "textwrap": textwrap
    }
    exec(func_code, global_env, local_env)
    return local_env[func_name]

def parse_final_output(raw_response: str) -> str:
    pattern = r'\[###START_FINAL_OUTPUT###\](.*?)\[###END_FINAL_OUTPUT###\]'
    match = re.search(pattern, raw_response, re.DOTALL)
    if not match:
        raise ValueError("Missing final-output markers")
    return match.group(1).strip()

def get_unit_test_repair_prompt(instruction: str, output: str) -> List[Dict[str, str]]:
    return [{
        "role": "user",
        "content": f"""You are tasked with validating and repairing an output based on the given instruction.

Instruction:
[###START_INSTRUCTION###]
{instruction}
[###END_INSTRUCTION###]

Output to verify and repair:
[###START_OUTPUT###]
{output}
[###END_OUTPUT###]

Guidelines:
- First, read the instruction and note every required directive or format.
- If the output already fully satisfies the instruction, do not modify it.
- Otherwise, make the minimum edits needed to fully satisfy every requirement.
- Preserve any correct parts; only fix what's broken.
- Do not include explanations or commentary.
- Enclose your final response inside [###START_FINAL_OUTPUT###] and [###END_FINAL_OUTPUT###] markers.

Example format:
[###START_FINAL_OUTPUT###]
(corrected_or_verified_output)
[###END_FINAL_OUTPUT###]"""
    }]

def get_negative_unit_test_repair_prompt(instruction: str, output: str) -> List[Dict[str, str]]:
    return [{
        "role": "user",
        "content": f"""You are tasked with validating and negatively repairing an output based on the given instruction.

Instruction:
[###START_INSTRUCTION###]
{instruction}
[###END_INSTRUCTION###]

Output to verify and negatively repair:
[###START_OUTPUT###]
{output}
[###END_OUTPUT###]

Guidelines:
- First, carefully read the instruction and understand the required directives or formatting.
- Then review the given output:
  - If the output **already violates** the instruction in a clear and deliberate way, leave it unchanged.
  - If the output **accidentally satisfies** the instruction or does not violate it clearly, modify it to explicitly break the instruction.
- When negatively repairing:
  - You must **insert forbidden elements** if the instruction prohibits them (e.g., add commas, change format, use disallowed styles).
  - You must **omit required structures** if the instruction enforces specific formatting (e.g., skip JSON wrapping, ignore punctuation rules).
  - Do not simply remove helpful content — focus on breaking the instruction's rules.
- Do not explain or justify your changes.
- Your final output must clearly and explicitly violate the instruction.
- Enclose your response inside [###START_FINAL_OUTPUT###] and [###END_FINAL_OUTPUT###].

Example format:
[###START_FINAL_OUTPUT###]
(explicitly instruction-breaking output)
[###END_FINAL_OUTPUT###]
"""
    }]
# ——— Core async validate & repair routine with history ———

async def validate_and_repair(
    instruction: str,
    output: str,
    checking_code: str,
    max_repair_attempts: int,
    semaphore: asyncio.Semaphore,
    stage:str
) -> Dict[str, Any]:
    """
    Validate the output with a dynamic checking function.
    If it fails, call LLM to repair up to max_repair_attempts times.
    Returns dict with history, attempts, success, and error if any.
    """
    history = [output]
    # compile checking function
    try:
        check_fn = load_checking_function_from_string(checking_code)
    except Exception as e:
        return {
            "history": history,
            "attempts": 0,
            "success": False,
            "error": f"compile error: {e}"
        }

    async def repair_once(text: str) -> str:
        if stage in ["directive_positive", "conflicting_directive_positive"]:
            prompt = get_unit_test_repair_prompt(instruction, text)
        elif stage in ["conflicting_directive_negative"]:
            prompt = get_negative_unit_test_repair_prompt(instruction, text)
        async with semaphore:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=prompt,
                temperature=0.2,
                max_tokens=2048
            )
        repaired = parse_final_output(resp.choices[0].message.content)
        history.append(repaired)
        return repaired

    import signal

    class TimeoutException(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutException("verifier function timed out")

    for attempt in range(max_repair_attempts + 1):
        # run check sync with timeout (Unix only)
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(5)  # 5 second timeout
            if stage in ["directive_positive", "conflicting_directive_negative"]: 
                is_valid = check_fn(history[-1])
            elif stage in ["conflicting_directive_positive"]:
                is_valid = not check_fn(history[-1])
            signal.alarm(0)  # Disable alarm
        except TimeoutException as e:
            print("Timeout!!!")
            return {
                "history": history,
                "attempts": attempt,
                "success": False,
                "error": f"checking error: {e}"
            }
        except Exception as e:
            return {
                "history": history,
                "attempts": attempt,
                "success": False,
                "error": f"checking error: {e}"
            }
        finally:
            signal.alarm(0)  # Always disable alarm

        if is_valid:
            return {"history": history, "attempts": attempt, "success": True}

        if attempt < max_repair_attempts:
            try:
                await repair_once(history[-1])
            except Exception as e:
                return {
                    "history": history,
                    "attempts": attempt + 1,
                    "success": False,
                    "error": f"repair error: {e}"
                }

    return {"history": history, "attempts": max_repair_attempts, "success": False}

# ——— Safe wrapper ———

async def process_entry(
    entry: Dict[str, Any],
    max_repair: int,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """
    For a single data entry, validate & repair both positive and negative units,
    then attach results into entry and return it.
    """
    checking_code = entry["verifier_fn"]
    instr_pos = entry["directive"]
    instr_neg = entry["conflicting_directive"]

    pos_units = entry["unit_test"]["positive_parsed_units"]
    conflicting_pos_units = entry["conflicting_unit_test"]["positive_parsed_units"]
    conflicting_neg_units = entry["conflicting_unit_test"]["negative_parsed_units"]

    # schedule all validation/repair tasks
    tasks = []
    for u in pos_units:
        tasks.append(validate_and_repair(instr_pos, u, checking_code, max_repair, semaphore, stage="directive_positive"))
    for u in conflicting_pos_units:
        tasks.append(validate_and_repair(instr_neg, u, checking_code, max_repair, semaphore, stage="conflicting_directive_positive"))
    for u in conflicting_neg_units:
        tasks.append(validate_and_repair(instr_neg, u, checking_code, max_repair, semaphore, stage="conflicting_directive_negative"))

    results = await asyncio.gather(*tasks)

    # split results and attach back into entry
    entry["positive_unit_test_results"] = results[: len(pos_units)]
    entry["conflicting_positive_unit_test_results"] = results[len(pos_units):len(pos_units)+len(conflicting_pos_units)]
    entry["conflicting_negative_unit_test_results"] = results[-len(conflicting_neg_units):]
    return entry

import ujson

def load_json(data_path):
    with open(data_path, "r", encoding='utf-8-sig') as f:
        data = ujson.load(f)
    return data

def load_line_json(data_path):
    data = []
    with open(data_path, "r") as f:
        for line in f:
            # print(line)
            data.append(ujson.loads(line))
    return data

def save_json(data, data_path):
    with open(data_path, "w") as f:
        ujson.dump(data, f, indent=2, ensure_ascii=False)

# ——— Resume support ———

def load_done_instructions(filepath: str) -> set:
    """
    Read existing output file lines to collect already processed instructions.
    """
    done = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                done.add(f'{d["directive"]}{d["verifier_fn"]}{d["conflicting_directive"]}')
    except FileNotFoundError:
        pass
    return done


async def main():
    inputs = load_line_json(PATH_TO_UNIT_TEST)
    done = load_done_instructions(PATH_TO_OUTPUT)
    sem = asyncio.Semaphore(THREAD_NUM)

    print(f"Already Process {len(done)}!!!!")

    # filter out already-done
    pending = [d for d in inputs if f'{d["directive"]}{d["verifier_fn"]}{d["conflicting_directive"]}' not in done]
    total = len(pending)
    completed = 0

    with open(PATH_TO_OUTPUT, "a", encoding="utf-8") as f:
        tasks = [
            asyncio.create_task(process_entry(entry, max_repair=2, semaphore=sem))
            for entry in pending
        ]

        for future in asyncio.as_completed(tasks):
            try:
                entry = await future
            except Exception as e:
                # skip this entry on error
                print(f"Entry error, skipping: {e}")
                completed += 1
                print(f"[{completed}/{total}]")
                continue

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            completed += 1
            print(f"[{completed}/{total}] Saved: {entry['directive']}")

    print(f"🎉 All done! {completed}/{total} entries processed.")


if __name__ == "__main__":
    asyncio.run(main())
