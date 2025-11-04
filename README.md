# [NeurIPS 2025] Beyond Oracle: Verifier-Supervision for Instruction Hierarchy in Reasoning and Instruction-Tuned LLMs
This repository implements the **end-to-end pipeline** proposed in our paper, covering:

1. **Prompt & Verifier Generation** — create scaffolds, directives, conflicting directives, and their verifier functions.  
2. **Automatic Unit-Test Generation** — produce tests that check data-quality constraints.  
3. **Filtering & Repair** — run the unit tests + verifiers to filter out bad instances and automatically repair failed ones.

Everything is fully asynchronous for high throughput with OpenAI APIs.

---
## TL;DR

We align instruction-tuned and reasoning LLMs on instruction hierarchy via executable verifier supervision, enabling oracle-free and trace-free training that generalizes to safety benchmarks.

## Abstract

Large language models (LLMs) are often prompted with multi-level directives, such as system instructions and user queries, that imply a hierarchy of authority. Yet models frequently fail to enforce this structure, especially in multi-step reasoning where errors propagate across intermediate steps. Existing methods rely on oracle completions but lack verifiable reward signals or intermediate traces, limiting their applicability. We introduce a unified supervision framework that embeds programmatically verifiable checkers into synthesized instruction-conflict instances. Each instance pairs a compliance directive with a conflicting one, along with an executable verifier that deterministically checks output adherence. This enables alignment without oracle labels or reasoning traces, supporting both instruction-tuned and reasoning models. The framework is instantiated via a synthesis pipeline that includes unit-test–based validation, LLM-assisted repair, and a probabilistic analysis of cleaning reliability. Fine-tuning on the resulting data improves instruction hierarchy adherence and boosts safety robustness, generalizing to adversarial safety benchmarks without task-specific supervision. This highlights verifiable supervision as a scalable foundation for robust alignment. All code, dataset, and verifier pipeline are publicly available at: https://github.com/cycraft-corp/BeyondOracle.

---
## Dataset

We provide the full training dataset used in the paper —  
**22,922 high-quality verifier-supervised examples** — located at:

```
./data/train_data.json
```

You can use this as a reference dataset, for fine-tuning, or for verifying the pipeline’s behavior.


## Repository Layout

```
.
├─ async_generate.py          # Step 1: prompts + verifiers
├─ async_gen_unit_test.py     # Step 2: unit-test generation
├─ async_data_cleaning.py     # Step 3: filtering & repair
├─ requirements.txt
└─ README.md
```

## Configuration

All scripts **have no CLI arguments**.  
Instead, open each file and edit the config section near the top:

| Variable         | Description                                      |
|------------------|--------------------------------------------------|
| `OPENAI_API_KEY` | Your OpenAI API key.  |
| `NUM_THREADS`    | Number of concurrent async workers               |
| `INPUT_PATH`     | Path to input file                               |
| `OUTPUT_PATH`    | Path to save results                             |

---

## Running the Full Pipeline

Run these scripts **in order**:

```
pip install -r requirements.txt
python async_generate.py       # ➜ generates prompts and verifiers
python async_gen_unit_test.py  # ➜ generates unit tests
python async_data_cleaning.py  # ➜ filters and repairs data
```

### Input / Output Dependency

Each script takes the output of the previous one as its input:

| Script                   | Input                                       | Output                     |
|--------------------------|---------------------------------------------|----------------------------|
| `async_generate.py`      | —                                           | `generated_prompts.jsonl`  |
| `async_gen_unit_test.py` | `generated_prompts.jsonl`                   | `unit_tests.jsonl`         |
| `async_data_cleaning.py` | `generated_prompts.jsonl` + `unit_tests.jsonl` | `clean_data.jsonl`      |

- `async_generate.py` creates the initial prompt scaffolds and verifier functions.
- `async_gen_unit_test.py` takes those prompts and generates unit tests to check data correctness.
- `async_data_cleaning.py` uses both the prompts and test results to filter and repair invalid or low-quality examples.

Make sure the input and output paths are consistent across scripts by editing the config section at the top of each file.`

### Stage Overview

| Step | Script                  | Description                                        | Paper Section | Output File                |
|------|-------------------------|----------------------------------------------------|----------------|----------------------------|
| 1️⃣   | `async_generate.py`       | Generate prompts and verifier functions            | §4.1            | `generated_prompts.jsonl`  |
| 2️⃣   | `async_gen_unit_test.py`  | Generate unit tests to verify data integrity       | §4.2             | `unit_tests.jsonl`         |
| 3️⃣   | `async_data_cleaning.py`  | Run tests + verifiers to filter and repair data    | §4.2             | `clean_data.jsonl`         |

## Citation
```
@inproceedings{
huang2025beyond,
title={Beyond Oracle: Verifier-Supervision for Instruction Hierarchy in Reasoning and Instruction-Tuned {LLM}s},
author={Sian-Yao Huang and Li-Hsien Chang and Che-Yu Lin and Cheng-Lin Yang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=IQ513IX1G5}
}
```
