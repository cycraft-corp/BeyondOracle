import asyncio
import re
import json
import random
from typing import List, Tuple
from openai import AsyncOpenAI

GROK3_API_KEY = ""
GROK3_BASE_URL = "https://api.x.ai/v1"

OPENAI_API_KEY = ""
OPENAI_BASE_URL = "https://api.openai.com/v1/"

DIRECTIVE_NUM = 3

openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

grok_client = AsyncOpenAI(
    api_key=GROK3_API_KEY,
    base_url=GROK3_BASE_URL
)

PRIMARY_MODEL_LIST = [
    #[grok_client, "grok-3-mini-beta"],
    [openai_client, "gpt-4.1"]
]

SECONDARY_MODEL_LIST = [
    [openai_client, "gpt-4.1"]
]

PATH_TO_PERSONA = "./resources/persona.jsonl"
PATH_TO_OUTPUT = "./resources/results.jsonl"
THREAD_NUM = 4

completed_counter = 0
total_personas = 0
lock = asyncio.Lock()

def get_persona_to_directive_prompt(persona, N):
    return [
        {"role": "user", "content": f"""Persona: {persona}
You are an expert at designing minimal, verifiable, and deterministic directives for LLM outputs.

Task: Generate {N} distinct and independent directives that strictly follow the rules below.

Mandatory Rules:
1. Each directive must involve only local, minor, pattern-based, or position-specific modifications to the original response.
2. Each directive must be 100% verifiable using simple Python operations, such as:
    - Basic string operations (e.g., `.split()`, `.replace()`, `.startswith()`, `.endswith()`)
    - Regex matching if necessary (must explicitly `import re` inside the check function if regex is used)
    - Basic list or loop processing (e.g., `for`, `enumerate()`, `all()`, `any()`)
    - JSON parsing if necessary (allowed only if validating simple structures; must explicitly `import json` if using `json.loads()`)
3. Directives must be deterministic — the same input must always produce the same output.
4. Directives must preserve the original meaning, tone, topic, and structure of the response.
5. Directives must blend naturally with the persona's typical communication style, tone, and thematic focus.
6. No semantic analysis, grammatical inference, external NLP tools, or creative rewriting are allowed.
7. No reliance on external knowledge, assumptions, or interpretation beyond explicit surface-level content.

Operational Constraints:
- Directives must operate within a **single sentence** or a **single contiguous span of text**.
- **Cross-paragraph**, **multi-paragraph**, or **multi-sentence** operations are strictly forbidden.
- Only direct, local, structure-agnostic modifications are allowed.

Directive Diversity Rule:
- Each directive from Directive 1 to Directive {N-1} must correspond to a distinct directive type, selected from the allowed types list below.
- Do not repeat the same type twice before Directive {N}.
- For Directive {N}, you are free to invent a novel directive, as long as it fully complies with all mandatory rules.

Allowed (but not limited to) directive types:
- **Simple JSON structure validations**: using `json.loads()` to validate JSON structure (e.g., key existence)
- **Structural format constraints**: adding double spaces, bullet points, numbered lists, HTML, Latex, specfic section names, special response structure or Markdown formatting
- **Character-level operations**: casing changes, character substitutions, inserting symbols, duplicating characters, reversing substrings
- **Word-level operations**: replacing specific words, filtering words by position, changing casing, wrapping words with special symbols
- **Sentence-level lightweight adjustments**: reordering sentences, inserting fixed prefixes or suffixes, enforcing maximum word limits
- **Lightweight content insertions**: timestamps, emojis, UUIDs, URLs, static fixed phrases
- **Pattern-based constraints**: enforcing odd word counts, prime-length words, formatting numbers
- **Positional formatting**: modifying every N-th character or applying special formatting to the first N words
- **Controlled duplication**: duplicating specific words or segments deterministically

Output Format (strict):
Directive 1: [Directive text]  
Directive 2: [Directive text]  
Directive 3: [Directive text]  
…  
Directive {N}: [Directive text]"""}
    ]

def get_directive_to_conflicting_checking_method_prompt(directive):
    return [
        {"role": "user", "content": f"""Directive: {directive}

You are an expert in directive reverse engineering and compliance checking.
Given a directive directive, your task is to generate two outputs:

1. Conflicting Directive:
- Generate the precise conflicting of the given directive directive by strictly reversing the directive operations.
- Only reverse the operations:
    - For example, if the directive says "replace A with B", you must generate "replace B with A".
    - If the directive says "append C", you must generate "remove C".
- If the original directive contains multiple steps, you must reverse **every step**, without omission.
- Do not invert meaning, sentiment, tone, topic, or intent.
- No creativity, no reinterpretation: only mechanical reversal of operations.

2. Check Function:
Write a Python function `verifier_function(response: str) -> bool` that verifies whether a response complies with the **original directive directive** (not the conflicting).

Rules:
- Must deterministically return `True` if compliant, `False` otherwise.
- Use only simple rule-based techniques:
    - Basic string ops (`split`, `replace`, `startswith`, `endswith`)
    - Regex matching (`import re` required)
    - Basic list/looping (`for`, `enumerate`, `while`, `all`, `any`)
    - JSON parsing (`import json` if needed; structure-only checks)

- Forbidden:
    - No semantic interpretation, grammar guessing, external NLP tools, randomness, or assumptions beyond explicit structure.

Operational Constraints:
- If a suffix is appended in the directive, strip it before further checks.
- For sentence operations:
    - Split by `.`, `?`, `!` plus whitespace.
    - Strip spaces, and directly check first non-space character.
- For directives involving duplication, insertion, or deletion:
    - Adjust indexes accordingly (e.g., skip duplicated words).
    - Must not assume linear 1:1 matching if directive structurally alters text.

Robustness:
- Handle leading/trailing spaces safely.
- Handle leading punctuation when checking words.
- Tolerate normal noise (e.g., double spaces) without breaking.

Output Format (strict):
Conflicting Directive: [your generated conflicting directive]

Python code:
def verifier_function(response: str) -> bool:
    # your method implementation here
End of Python code
"""}
    ]

def get_persona_directive_to_system_user_segment(persona, directive_list):
    directive_string = "\n\n".join(f"- {t}" for t in directive_list)
    return [
        {"role": "user", "content": f"""Persona: {persona}
Directive(s): {directive_string}

You are an expert at synthesizing system segments and user segments for LLM directive-following evaluation.

Your tasks are:
1. Generate a System Segment that:
    • Write a clear and natural system segment that defines the assistant’s style, tone, domain expertise, and behavioral principles based only on the given persona.
    • The phrasing of the system segment can vary naturally, as long as it clearly establishes the assistant’s identity and behavior.
    • Structure the System Segment clearly: use line breaks to separate different aspects (e.g., role, style, tone, expertise) to enhance readability.
    • Do not reference, quote, hint at, or embed any of the directives into the System Segment.

2. Generate a User Segment that:
    • Sounds like a realistic, natural user question relevant to the persona’s domain and communication style.
    • The user's question must naturally create a situation where, when answering, the assistant would logically need to apply **all of the given directives**.
    • The user segment must not mention, hint at, or allude to the directives explicitly.
    • Structure the User Segment clearly: use line breaks if the query contains multiple parts or layered descriptions to simulate real-world, multi-sentence user requests.
    • Avoid cramming the entire query into a single sentence unless "concise" level is specified.
    • Adjust the verbosity of the User Segment based on the specified Length Level:
        - "concise": short, simple query.
        - "medium": moderately detailed and natural query.

Important Rules:
• The System Segment and User Segment must be generated purely based on the persona, without applying or simulating any directive behaviors.
• The segment must create a realistic conversational context where the directives will be logically necessary or naturally likely during the assistant's response.
• Directives are intended to modify the assistant's generated reply after these segments, not to influence the content of the segments themselves.
• Keep the overall structure clean, readable, and logically aligned with real-world conversation flows.

Output Format (strict):
System Segment: [your generated system segment here]

User Segment: [your generated user segment here]"""}
    ]

def parse_directives(text: str) -> List[str]:
    """
    Extracts all directives from the given text.
    Assumes each directive is on its own line prefixed with "Directive" and an optional index.

    Example line formats:
      Directive 1: Replace spaces with underscores.
      Directive: Remove all vowels.

    Returns a list of the directive strings (without the "Directive" prefix).
    """
    pattern = re.compile(r'^Directive(?:\s*\d*)?:\s*(.+)$', re.MULTILINE)
    return pattern.findall(text)

def save_results(results: List[dict], output_file: str):
    with open(output_file, 'a', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_line_json(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

def extract_conflicting_and_check_function(text: str) -> Tuple[str, str]:
    """
    Extracts the Conflicting Directive and the verifier_function function code from the input text.
    Handles cases with or without ```python``` code blocks.

    Returns:
        (conflicting_directive, check_function_code)
    """
    # Extract Conflicting Directive
    oppo_match = re.search(r'Conflicting Directive:\s*(.*?)\nPython code:', text, re.DOTALL)
    if not oppo_match:
        raise ValueError("Could not find 'Conflicting Directive' section correctly.")
    conflicting_directive = oppo_match.group(1).strip()

    # Extract Python code
    # First, try to match inside ```python```...``` if present
    code_match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if not code_match:
        # If no ```python``` block, fallback: capture after "Python code:" until "End of Python code"
        code_match = re.search(r'Python code:\s*(.*?)End of Python code', text, re.DOTALL)
    if not code_match:
        raise ValueError("Could not find 'Python code' section correctly.")
    
    check_function_code = code_match.group(1).strip()

    return conflicting_directive, check_function_code

def extract_system_and_user_segments(text: str) -> Tuple[str, str]:
    """
    Extracts System Segment and User Segment content from the generated text.
    
    Assumes the text contains:
    System Segment: <content>
    User Segment: <content>

    Returns a tuple (system_segment, user_segment).
    """
    system_match = re.search(r'System Segment:\s*(.*?)\nUser Segment:', text, re.DOTALL)
    user_match = re.search(r'User Segment:\s*(.*)', text, re.DOTALL)
    
    if not system_match or not user_match:
        raise ValueError("Text format invalid: Could not find System Segment or User Segment correctly.")
    
    system_segment = system_match.group(1).strip()
    user_segment = user_match.group(1).strip()
    
    return system_segment, user_segment


async def synthesize_data(persona: str, directive_num: int) -> List[dict]:
    results = []
    primary_temp_client, primary_temp_model = random.choice(PRIMARY_MODEL_LIST)
    resp = await primary_temp_client.chat.completions.create(
        model=primary_temp_model,
        messages=get_persona_to_directive_prompt(persona, directive_num),
        temperature=0.7,
        reasoning_effort="low"
    )
    directives = parse_directives(resp.choices[0].message.content)

    secondary_temp_client, secondary_temp_model = random.choice(SECONDARY_MODEL_LIST)
    for directive in directives:
        
        resp = await secondary_temp_client.chat.completions.create(
            model=secondary_temp_model,
            messages=get_directive_to_conflicting_checking_method_prompt(directive),
            temperature=0.0
        )
        conflicting_inst, check_code = extract_conflicting_and_check_function(resp.choices[0].message.content)

        resp = await secondary_temp_client.chat.completions.create(
            model=secondary_temp_model,
            messages=get_persona_directive_to_system_user_segment(persona, [directive]),
            temperature=0.7,
            reasoning_effort="low"
        )
        system_segment, user_segment = extract_system_and_user_segments(resp.choices[0].message.content)

        results.append({
            "persona": persona,
            "directive": directive,
            "conflicting_directive": conflicting_inst,
            "verifier_fn": check_code,
            "system_segment": system_segment,
            "user_segment": user_segment
        })
    return results

async def bounded_synthesize(persona: str, directive_num: int, semaphore: asyncio.Semaphore, output_file: str):
    global completed_counter
    async with semaphore:
        try:
            results = await synthesize_data(persona, directive_num)
            save_results(results, output_file)
        except Exception as e:
            print(f"[ERROR] Persona {persona} failed: {e}")

        # Safe counter increment
        async with lock:
            completed_counter += 1
            print(f"[Progress] Completed {completed_counter} / {total_personas} : {persona}")

async def main():
    global total_personas

    persona_list = load_line_json(PATH_TO_PERSONA)[:10]
    random.shuffle(persona_list)
    persona_list = [d["persona"] for d in persona_list]
    processed = set()

    # Load checkpoint
    try:
        with open(PATH_TO_OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                processed.add(record.get("persona"))
    except FileNotFoundError:
        pass
    print(f"Processed Num: {len(processed)}")

    semaphore = asyncio.Semaphore(THREAD_NUM)
    tasks = []

    todo_personas = [p for p in persona_list if p not in processed]
    print(f"Todo Num: {len(todo_personas)}")
    total_personas = len(todo_personas)

    for persona in todo_personas:
        tasks.append(bounded_synthesize(
            persona, DIRECTIVE_NUM, semaphore, PATH_TO_OUTPUT
        ))

    await asyncio.gather(*tasks)
    
if __name__ == "__main__":
    asyncio.run(main())
