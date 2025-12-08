# =========================================================
# HF Evaluation Script – Zero-Shot & Few-Shot auf HuggingFace-Datasets
# MIT vollständigem Prompt-Logging für jede Anfrage
# =========================================================

from __future__ import annotations

import os
import json
import random
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from datasets import load_dataset as hf_load_dataset
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

# =========================================================
# 🔧 KONFIGURATION
# =========================================================

DEBUG_LOG_PROMPTS = True
DEBUG_PROMPT_DIR = Path("debug_prompts")

EVAL_MODE = "few-shot"  # "zero-shot" oder "few-shot"
RANDOM_SEED = 42

HF_DATASET_NAME = "tasksource/bigbench"
HF_TASK_NAME = "logical_fallacy_detection"

ZERO_SHOT_EVAL_SPLITS = {
    "validation": 10,
}

FEW_SHOT_TRAIN_SPLIT = "train"
FEW_SHOT_TRAIN_MAX = 200
FEW_SHOT_CONTEXT_EXAMPLES = 10

FEW_SHOT_TEST_SPLITS = {
    "validation": 10,
}

MODEL_NAME = "gpt-4o-mini"

ZERO_SHOT_PROMPT = (
    'You MUST answer with exactly one string: either "Valid" or "Invalid". '
    'Do not output explanations, reasoning steps, punctuation, or additional text. '
    '"Valid" = the argument is logically valid. '
    '"Invalid" = the conclusion does not necessarily follow from the premises.\n\n'
    'Now answer the following question with exactly one word: "Valid" or "Invalid".'
)

FEW_SHOT_BASE_SYSTEM_PROMPT = (
    'You classify arguments as "Valid" or "Invalid". '
    '"Valid" = the argument is logically valid. '
    '"Invalid" = the conclusion does not necessarily follow from the premises.\n\n'
)

RESULTS_ROOT = Path("hf_eval_results_new")


# =========================================================
# Logging
# =========================================================

def log_header(title: str):
    print("\n" + "=" * 80)
    print(f"🔵 {title}")
    print("=" * 80)

def log_sub(title: str):
    print(f"\n--- {title} ---")

def log_line(msg: str):
    print(f"  • {msg}")


# =========================================================
# Data Classes
# =========================================================

@dataclass
class Example:
    input: str
    label: str
    choices: Optional[List[str]] = None

@dataclass
class EvalRunResult:
    mode: str
    model: str
    base_system_prompt: Optional[str] = None
    used_prompt: Optional[str] = None
    few_shot_examples: List[Example] = field(default_factory=list)
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    accuracy: float = 0.0


# =========================================================
# OpenAI Client
# =========================================================

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Bitte OPENAI_API_KEY setzen.")
    return OpenAI(api_key=api_key)


# =========================================================
# Normalisierung & Alignment
# =========================================================

def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    if t.endswith((".", "!", "?")):
        t = t[:-1]
    t = t.strip().lower()
    if t.startswith("choice:"):
        t = t[len("choice:"):].strip()
    bool_map = {
        "true": "bool_true",
        "yes": "bool_true",
        "y": "bool_true",
        "false": "bool_false",
        "no": "bool_false",
        "n": "bool_false",
    }
    return bool_map.get(t, t)

def align_model_answer(answer_raw: str, gold_raw: str, choices=None) -> str:
    if answer_raw is None:
        return ""
    t = answer_raw.strip()
    if t.lower().startswith("choice:"):
        t = t.split(":", 1)[1].strip()
    if gold_raw and t == gold_raw.strip():
        return gold_raw.strip()
    if choices:
        norm_choices = [c.strip() for c in choices]
        if t in norm_choices:
            return t
        toks = t.replace(",", " ").replace(".", " ").split()
        for tok in toks:
            if tok in norm_choices:
                return tok
    return t

# =========================================================
# PROMPT LOGGER – vollständiger Prompt pro Anfrage
# =========================================================

def log_prompt_to_file(messages, mode, split_name, idx):
    if not DEBUG_LOG_PROMPTS:
        return

    DEBUG_PROMPT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file = (
        DEBUG_PROMPT_DIR /
        f"{HF_TASK_NAME}_{mode}_{split_name}_idx{idx}_{timestamp}.json"
    )

    payload = {
        "task_name": HF_TASK_NAME,
        "mode": mode,
        "split": split_name,
        "index": idx,
        "messages": messages,
    }

    file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    log_line(f"🔍 Prompt geloggt: {file}")


# =========================================================
# HF Dataset Loader
# =========================================================

def load_hf_bigbench_dataset(task_name, split, max_examples):
    if max_examples <= 0:
        log_line(f"⚠️ Split '{split}' übersprungen (0 Beispiele).")
        return []

    ds = hf_load_dataset(HF_DATASET_NAME, task_name, split=split)

    available = len(ds)
    if max_examples > available:
        max_examples = available

    ds = ds.select(range(max_examples))

    out = []
    for item in ds:
        inp = item.get("input") or item.get("inputs") or item.get("question")
        label = (
            item.get("label")
            or (item.get("targets")[0] if item.get("targets") else None)
        )
        choices = item.get("multiple_choice_targets") or item.get("choices")
        if not inp or not label:
            continue
        out.append(Example(str(inp), str(label), choices))
    return out


# =========================================================
# ZERO-SHOT Evaluation
# =========================================================

def run_zero_shot_on_dataset(prompt, dataset, split_name, model=MODEL_NAME, system_message=None):
    client = get_client()
    results = []
    correct = 0

    for idx, ex in enumerate(dataset):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": f"{prompt}\n\n{ex.input}"})

        # PROMPT-LOG HIER
        log_prompt_to_file(messages, "zero-shot", split_name, idx)

        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0
        )

        raw = resp.choices[0].message.content.strip()
        aligned = align_model_answer(raw, ex.label, ex.choices)
        norm = normalize_answer(aligned)
        gold_norm = normalize_answer(ex.label)

        ok = norm == gold_norm
        if ok:
            correct += 1

        results.append({
            "Index": idx,
            "Input": ex.input,
            "correctAnswer": ex.label,
            "MbaseAnswer": aligned,
            "MbaseAnswerRaw": raw,
            "judgement": ok,
        })

    return EvalRunResult(
        mode="zero-shot",
        model=model,
        used_prompt=prompt,
        test_results=results,
        accuracy=correct / len(dataset),
    )


# =========================================================
# FEW-SHOT Evaluation
# =========================================================

def build_few_shot_block(train, n):
    return train[:n]

def format_few_shot_block(block):
    return "\n\n".join(
        f"Example {i}:\nQuestion: {ex.input}\nAnswer: {ex.label}"
        for i, ex in enumerate(block, start=1)
    )

def chat_with_retry(client, **kwargs):
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            time.sleep(1.0 * (2**attempt))
    raise RuntimeError("Rate limits nach 5 Versuchen")

def evaluate_few_shot(train_dataset, test_dataset, base_prompt,
                      split_name, model=MODEL_NAME,
                      max_train_examples=FEW_SHOT_CONTEXT_EXAMPLES):

    client = get_client()

    block = build_few_shot_block(train_dataset, max_train_examples)
    block_text = format_few_shot_block(block)

    system_prompt = (
        base_prompt +
        "\n\nAdditionally I am giving you the following examples to learn from:\n\n" +
        block_text +
        "\n\nLearn from the given examples and answer the question."
    )

    # gesamten Few-Shot Prompt 1× loggen
    Path("debug_prompts").mkdir(exist_ok=True)
    Path("debug_prompts/fewshot_system_prompt.txt").write_text(system_prompt, "utf-8")

    results = []
    correct = 0

    for idx, ex in enumerate(test_dataset):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex.input},
        ]

        # PROMPT-LOG HIER
        log_prompt_to_file(messages, "few-shot", split_name, idx)

        resp = chat_with_retry(client, model=model,
                               messages=messages, temperature=0.0)

        raw = resp.choices[0].message.content.strip()
        aligned = align_model_answer(raw, ex.label, ex.choices)
        norm = normalize_answer(aligned)
        gold_norm = normalize_answer(ex.label)

        ok = norm == gold_norm
        if ok:
            correct += 1

        results.append({
            "Index": idx,
            "Input": ex.input,
            "correctAnswer": ex.label,
            "MbaseAnswer": aligned,
            "MbaseAnswerRaw": raw,
            "judgement": ok,
        })

    return EvalRunResult(
        mode="few-shot",
        model=model,
        base_system_prompt=base_prompt,
        used_prompt=system_prompt,
        few_shot_examples=block,
        test_results=results,
        accuracy=correct / len(test_dataset),
    )

# =========================================================
# SAVE JSON
# =========================================================

def save_run_to_json(path, *, dataset_path, dataset_size, config, result):
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "path": dataset_path,
            "num_examples": dataset_size,
        },
        "config": config,
        "result": {
            "mode": result.mode,
            "prompt": result.used_prompt,
            "accuracy": result.accuracy,
            "results": result.test_results,
        },
    }

    if result.mode == "few-shot":
        payload["dataset"]["few_shot_examples"] = len(result.few_shot_examples)
        payload["config"]["base_system_prompt"] = result.base_system_prompt

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
    log_header(f"Saved: {path}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    system_message = None

    root = RESULTS_ROOT / f"{HF_TASK_NAME}_{EVAL_MODE}_new"

    # ZERO-SHOT ---------------------------------------------------------------
    if EVAL_MODE == "zero-shot":

        config_base = {
            "prompt": ZERO_SHOT_PROMPT,
            "model_name": MODEL_NAME,
            "mode": "zero-shot",
        }

        for split_name, max_n in ZERO_SHOT_EVAL_SPLITS.items():
            log_header(f"[ZERO-SHOT] Split '{split_name}'")

            dataset = load_hf_bigbench_dataset(HF_TASK_NAME, split_name, max_n)
            if not dataset:
                log_line("⚠️ Keine Daten – skip.")
                continue

            result = run_zero_shot_on_dataset(
                ZERO_SHOT_PROMPT,
                dataset,
                split_name,
                MODEL_NAME,
                system_message,
            )

            out_dir = root / split_name
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = out_dir / f"zeroshot_{HF_TASK_NAME}_{split_name}_{ts}.json"

            save_run_to_json(
                out,
                dataset_path=f"{HF_DATASET_NAME}:{HF_TASK_NAME}[{split_name}]",
                dataset_size=len(dataset),
                config={
                    **config_base,
                    "split": split_name,
                    "used_examples": len(dataset),
                },
                result=result,
            )

    # FEW-SHOT ---------------------------------------------------------------
    elif EVAL_MODE == "few-shot":

        config_base = {
            "base_system_prompt": FEW_SHOT_BASE_SYSTEM_PROMPT,
            "model_name": MODEL_NAME,
            "mode": "few-shot",
        }

        log_header("[FEW-SHOT] Lade Trainingsdaten")
        train_data = load_hf_bigbench_dataset(
            HF_TASK_NAME, FEW_SHOT_TRAIN_SPLIT, FEW_SHOT_TRAIN_MAX
        )

        for split_name, max_n in FEW_SHOT_TEST_SPLITS.items():
            log_header(f"[FEW-SHOT] Test Split '{split_name}'")

            test_data = load_hf_bigbench_dataset(HF_TASK_NAME, split_name, max_n)
            if not test_data:
                log_line("⚠️ Keine Testdaten – skip.")
                continue

            result = evaluate_few_shot(
                train_dataset=train_data,
                test_dataset=test_data,
                base_system_prompt=FEW_SHOT_BASE_SYSTEM_PROMPT,
                split_name=split_name,
            )

            out_dir = root / split_name
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = out_dir / f"fewshot_{HF_TASK_NAME}_{split_name}_{ts}.json"

            save_run_to_json(
                out,
                dataset_path=f"{HF_DATASET_NAME}:{HF_TASK_NAME}[{split_name}]",
                dataset_size=len(test_data),
                config={
                    **config_base,
                    "train_split": FEW_SHOT_TRAIN_SPLIT,
                    "test_split": split_name,
                    "used_train_examples": len(train_data),
                    "used_test_examples": len(test_data),
                },
                result=result,
            )

    else:
        raise ValueError("EVAL_MODE muss 'zero-shot' oder 'few-shot' sein.")
