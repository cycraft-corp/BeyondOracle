import asyncio
import json
import re
from typing import List, Dict, Any
from openai import AsyncOpenAI

MODEL = ""
OPENAI_API_KEY = ""

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
)

PATH_TO_DATA = "./resources/results.jsonl"
PATH_TO_OUTPUT = "./resources/unit_test.jsonl"

THREAD_NUM = 4

def parse_multiple_llm_units(raw_output: str) -> List[str]:
    pattern = r'\[###START_OUTPUT###\](.*?)\[###END_OUTPUT###\]'
    matches = re.findall(pattern, raw_output, re.DOTALL)
    return [match.strip() for match in matches if match.strip()]

# Positive Prompt
def get_positive_unit_test_generation_prompt(instruction: str, unit_num: int) -> List[Dict[str, str]]:
    return [{
        "role": "user",
        "content": f"""You are tasked with generating {unit_num} independent positive example outputs for the following instruction.

:
[###START_INSTRUCTION###]
{instruction}
[###END_INSTRUCTION###]

Guidelines:
- Each output must strictly and independently satisfy the instruction in both **content** and **format**.
- Every specific directive or constraint mentioned in the instruction must be correctly and fully applied, with no omissions or mistakes.
- If the instruction specifies any required formatting (such as JSON structure, punctuation rules, or style), your outputs must precisely match the required structure without deviation.
- For word count requirements, interpret "words" as space-separated English words, not characters or tokens.
- Do not partially fulfill the instruction. Even small errors or missing directives are unacceptable.
- Do not explain, comment, or number your examples.
- Enclose each output separately inside [###START_OUTPUT###] and [###END_OUTPUT###] markers.
- Please create a short and coherent sentence, but avoid simply saying something like "This is a compliant sentence."

Example format:
[###START_OUTPUT###]
[example_1_content]
[###END_OUTPUT###]
[###START_OUTPUT###]
[example_2_content]
[###END_OUTPUT###]
..."""
    }]

# Negative Prompt
def get_negative_unit_test_generation_prompt(instruction: str, unit_num: int) -> List[Dict[str, str]]:
    return [{
        "role": "user",
        "content": f"""You are tasked with generating {unit_num} independent negative example outputs for the following instruction.

:
[###START_INSTRUCTION###]
{instruction}
[###END_INSTRUCTION###]

Guidelines:
- Each output must deliberately and explicitly violate the instruction.
- You must actively perform actions that are explicitly forbidden by the instruction.
- If the instruction prohibits the use of certain elements (e.g., commas, specific styles, formatting), your outputs must intentionally **insert** the prohibited elements into your response.
- If the instruction requires adherence to a specific structure (e.g., JSON formatting, markdown headings, punctuation rules), your outputs must **intentionally break or omit** the required structure.
- Do not copy, repeat, or paraphrase the instruction itself.
- Do not explain, comment, or number your examples.
- Enclose each output separately inside [###START_OUTPUT###] and [###END_OUTPUT###] markers.
- Please create a short and coherent sentence, but avoid simply saying something like "This is a compliant sentence."

Example format:
[###START_OUTPUT###]
[example_1_content]
[###END_OUTPUT###]
[###START_OUTPUT###]
[example_2_content]
[###END_OUTPUT###]
..."""
    }]


async def generate_both_for(
    instruction: str,
    unit_num: int,
    semaphore: asyncio.Semaphore,
    instruction_type: str
) -> Dict[str, Any]:
    async with semaphore:
        if instruction_type == "directive":
            positive_prompt = get_positive_unit_test_generation_prompt(instruction, unit_num)
            positive_response = await client.chat.completions.create(model=MODEL, messages=positive_prompt, temperature=0.0, max_tokens=3096)

            positive_raw_output = positive_response.choices[0].message.content
            negative_raw_output = ""

            positive_parsed_units = parse_multiple_llm_units(positive_raw_output)
            negative_parsed_units = []

        elif instruction_type == "conflicting_directive":
            negative_prompt = get_negative_unit_test_generation_prompt(instruction, unit_num)
            negative_response = await client.chat.completions.create(model=MODEL, messages=negative_prompt, temperature=0.0, max_tokens=3096)

            negative_raw_output = negative_response.choices[0].message.content
            negative_parsed_units = parse_multiple_llm_units(negative_raw_output)

            positive_prompt = get_positive_unit_test_generation_prompt(instruction, unit_num)
            positive_response = await client.chat.completions.create(model=MODEL, messages=positive_prompt, temperature=0.0, max_tokens=2048)

            positive_raw_output = positive_response.choices[0].message.content
            positive_parsed_units = parse_multiple_llm_units(positive_raw_output)
        else:
            print("!!!!!!")

    return {
        "unit_num": unit_num,
        "positive_raw_output": positive_raw_output,
        "positive_parsed_units": positive_parsed_units,
        "negative_raw_output": negative_raw_output,
        "negative_parsed_units": negative_parsed_units
    }
    
def load_line_json(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

def load_done_instructions(filepath: str) -> set:
    done = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                done.add(f'{data["directive"]}{data["verifier_fn"]}{data["conflicting_directive"]}')
    except FileNotFoundError:
        pass
    return done

async def process_instruction(
    inp,
    idx: int,
    total: int,
    semaphore: asyncio.Semaphore
):
    try:
        tasks = [
            generate_both_for(inp["directive"], 2, semaphore, "directive"),
            generate_both_for(inp["conflicting_directive"], 2, semaphore, "conflicting_directive"),
        ]
        unit_tests = await asyncio.gather(*tasks)
        record = {
            "directive": inp["directive"],
            "conflicting_directive": inp["conflicting_directive"],
            "verifier_fn": inp["verifier_fn"],
            "unit_test": unit_tests[0],
            "conflicting_unit_test": unit_tests[1]
        }

        with open(PATH_TO_OUTPUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        print(f"[{idx}/{total}] done: {inp['directive'][:50]}...")
    except Exception as e:
        print(e)
        print("!!!!")

async def main():
    inputs = load_line_json(PATH_TO_DATA)

    done = load_done_instructions(PATH_TO_OUTPUT)
    print(f"Resuming: {len(done)} instructions already completed.")

    semaphore = asyncio.Semaphore(THREAD_NUM)
    tasks = []
    total = len(inputs)
    for idx, inp in enumerate(inputs, start=1):
        if f'{inp["directive"]}{inp["verifier_fn"]}{inp["conflicting_directive"]}' in done:
            print(f"[{idx}/{total}] skip: already done.")
            continue
        tasks.append(process_instruction(inp, idx, total, semaphore))
    print(f"Total Task Num to Process: {len(tasks)}")
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
