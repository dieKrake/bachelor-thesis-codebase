"""
HF Evaluation Script – Zero-Shot, Few-Shot & RAG auf HuggingFace-Datasets
(z.B. tasksource/bigbench), mit vollständigem Prompt-Logging.

Modi (EVAL_MODE):
- "zero-shot":
    * Nur Val/Test-Splits, kein Training-Kontext.
    * Prompt wird direkt vor die Frage gepackt (User-Message).
- "few-shot":
    * Train-Split liefert Few-Shot-Beispiele (Q/A-Paare).
    * Diese werden in den System-Prompt eingebettet.
- "rag":
    * HF-Dataset als Testdaten.
    * Kontext kommt aus Azure Search (Index frei konfigurierbar).
    * Kontext wird wie Few-Shot-Beispiele in den System-Prompt eingebettet.

Alle Modi:
- Lesen ein gelabeltes Dataset (Input + Label [+ optional choices]) direkt aus HF.
- Robust gegenüber:
    * True/False
    * Yes/No
    * Multiple-Choice (multiple_choice_targets + targets)
- Speichern Accuracy + Einzelergebnisse als JSON mit Timestamp & Dataset-Info.
"""

from __future__ import annotations

import os
import json
import random
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

import requests
from datasets import load_dataset as hf_load_dataset
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

# =========================================================
# 🔧 KONFIGURATION
# =========================================================

# Debug: komplette Prompts pro Anfrage loggen?
DEBUG_LOG_PROMPTS = False
DEBUG_PROMPT_DIR = Path("debug_prompts")

# Modus wählen: "zero-shot", "few-shot" oder "rag"
EVAL_MODE = "zero-shot"  # "zero-shot" | "few-shot" | "rag"
RANDOM_SEED = 42

# HuggingFace Dataset-Konfiguration
HF_DATASET_NAME = "google/boolq"
HF_TASK_NAME = None

# ---------------- ZERO-SHOT ----------------
# Splits & Größen für Zero-Shot: split_name -> max_examples
ZERO_SHOT_EVAL_SPLITS: Dict[str, int] = {
    "validation": 200,
    # "test": 0,
}

# ---------------- FEW-SHOT -----------------
FEW_SHOT_TRAIN_SPLIT = "train"        # aus diesem Split kommen die Few-Shot-Beispiele
FEW_SHOT_TRAIN_MAX = 100              # wie viele Train-Beispiele maximal laden
FEW_SHOT_CONTEXT_EXAMPLES = 100        # wie viele davon tatsächlich im Prompt landen

FEW_SHOT_TEST_SPLITS: Dict[str, int] = {
    "validation": 200,
    # "test": 0,
}

# ---------------- RAG ---------------------
# Splits, die mit RAG evaluiert werden sollen
RAG_EVAL_SPLITS: Dict[str, int] = {
    "validation": 200,
    # "test": 0,
}

# Azure Search Konfiguration (per .env oder hier direkt)
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX = "boolq_train_100"  # <== HIER deinen Indexnamen eintragen
RAG_TOP_K = 3  # wie viele Dokumente aus Azure pro Anfrage holen

# ---------------- MODEL & PROMPTS ---------

MODEL_NAME = "gpt-4o-mini"

# Navigate
# zero_shot_prompt = (
#     'You MUST answer with exactly one string: either "True" or "False". '
#     'Do not output explanations, punctuation, or additional text. '
#     '"True" = you return to the starting point. '
#     '"False" = you do not return to the starting point.\n\n'
#     'Now answer the following question with "True" or "False":'
# )

# Elementary
# zero_shot_prompt = (
#     "You are solving multiple-choice math questions.\n"
#     "Each question lists several answer options, each prefixed with 'choice:'.\n"
#     "Read the question and the choices carefully.\n"
#     "Answer with exactly the text of the correct choice, without the 'choice:' prefix and without any explanation or extra text."
#   )

# Logical_Fallacy
# zero_shot_prompt = (
#   'You MUST answer with exactly one string: either "Valid" or "Invalid". '
#   'Do not output explanations, reasoning steps, punctuation, or additional text. '
#   '"Valid" = the argument is logically valid. '
#   '"Invalid" = the conclusion does not necessarily follow from the premises.\n\n'
#   'Now answer the following question with exactly one word: "Valid" or "Invalid".'
# )

# boolq
# zero_shot_prompt = (
#     'You MUST answer with exactly one string: either "True" or "False". '
#     'Do not output explanations, reasoning steps, or additional text.\n\n'
#     '"True" = The passage implies that the answer to the question is YES.\n'
#     '"False" = The passage implies that the answer to the question is NO, '
#     'or the passage does NOT provide enough evidence to answer YES.\n\n'
#     'Now read the following question and its passage, and answer with "True" or "False":'
# )

# Zero-Shot-Prompt – als User-Prompt vor der Frage
ZERO_SHOT_PROMPT = (
   "You MUST answer with exactly one string: either 'True' or 'False'. Do not output explanations, reasoning steps, or additional text.\n\n'True' = The passage provides explicit evidence that directly supports the question.\n'False' = The passage does not provide explicit evidence, implies the opposite, or lacks sufficient information to answer the question.\n\nNow read the following question and its passage, and answer with 'True' or 'False':"
)


# Few-Shot Basis-System-Prompt – in den System-Prompt, Beispiele werden angehängt
FEW_SHOT_BASE_SYSTEM_PROMPT = (
    'You MUST answer with exactly one string: either "True" or "False". '
    'Do not output explanations, reasoning steps, or additional text.\n\n'
    '"True" = The passage implies that the answer to the question is YES.\n'
    '"False" = The passage implies that the answer to the question is NO, '
    'or the passage does NOT provide enough evidence to answer YES.\n\n'
    'Now read the following question and its passage, and answer with "True" or "False":'
)

# RAG kann denselben Basis-Prompt nutzen (oder einen anderen, falls gewünscht)
RAG_BASE_SYSTEM_PROMPT = FEW_SHOT_BASE_SYSTEM_PROMPT

# Ergebnis-Ordner (Basis)
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
    mode: str           # "zero-shot" | "few-shot" | "rag"
    model: str
    base_system_prompt: Optional[str] = None
    used_prompt: Optional[str] = None       # bei few-shot/rag: System-Prompt (ohne Kontext je Beispiel)
    few_shot_examples: List[Example] = field(default_factory=list)
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    accuracy: float = 0.0


# =========================================================
# OpenAI Client + Retry
# =========================================================

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Bitte OPENAI_API_KEY setzen.")
    return OpenAI(api_key=api_key)


def chat_with_retry(client: OpenAI, **kwargs):
    """
    Wrapper um client.chat.completions.create mit einfachem Retry bei Rate-Limit.
    Nutzt exponentielles Backoff.
    """
    max_retries = 5
    base_delay = 1.0  # Sekunden

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            print(
                f"⏳ Rate limit erreicht (Versuch {attempt+1}/{max_retries}), "
                f"warte {wait:.1f}s..."
            )
            time.sleep(wait)


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
        norm_choices = [str(c).strip() for c in choices]
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

def log_prompt_to_file(
    messages: List[Dict[str, str]],
    mode: str,
    split_name: str,
    idx: int,
):
    if not DEBUG_LOG_PROMPTS:
        return

    DEBUG_PROMPT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file = (
        DEBUG_PROMPT_DIR
        / f"{HF_TASK_NAME}_{mode}_{split_name}_idx{idx}_{timestamp}.json"
    )

    payload = {
        "task_name": HF_TASK_NAME,
        "mode": mode,
        "split": split_name,
        "index": idx,
        "messages": messages,
    }

    file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_line(f"🔍 Prompt geloggt: {file}")


# =========================================================
# HF Dataset Loader
# =========================================================

def load_hf_dataset(
    split: str,
    max_examples: int,
) -> List[Example]:
    """
    Allgemeiner HF-Loader:
    - Für Datasets mit Subtask (z.B. tasksource/bigbench + HF_TASK_NAME)
    - Für Datasets ohne Subtask (z.B. google/boolq, HF_TASK_NAME=None)
    """
    if max_examples <= 0:
        log_line(f"⚠️ Split '{split}' übersprungen (0 Beispiele).")
        return []

    # Unterscheide: mit / ohne Task-Name
    if HF_TASK_NAME in (None, "", "None"):
        log_sub(f"Lade HF-Dataset: {HF_DATASET_NAME}[{split}]")
        ds = hf_load_dataset(HF_DATASET_NAME, split=split)
    else:
        log_sub(f"Lade HF-Dataset: {HF_DATASET_NAME}/{HF_TASK_NAME}[{split}]")
        ds = hf_load_dataset(HF_DATASET_NAME, HF_TASK_NAME, split=split)

    available = len(ds)
    if max_examples > available:
        log_line(
            f"⚠️ Gewünschte Größe {max_examples} > verfügbare Beispiele {available} "
            f"in split '{split}' – verwende {available} Beispiele."
        )
        max_examples = available
    else:
        log_line(f"Verwende {max_examples} von {available} verfügbaren Beispielen.")

    ds = ds.select(range(max_examples))

    out: List[Example] = []
    for item in ds:
        # Spezieller Pfad für BoolQ
        if HF_DATASET_NAME == "google/boolq":
            # BoolQ: question (str), passage (str), answer (bool)
            question = item["question"]
            passage = item["passage"]
            answer = item["answer"]

            inp = f"Question: {question}\nPassage: {passage}"
            label = "True" if bool(answer) else "False"
            choices = None

        # Fallback: generische BIG-Bench-Logik
        else:
            inp = (
                item.get("input")
                or item.get("inputs")
                or item.get("question")
                or item.get("instruction")
                or None
            )
            label = (
                item.get("label")
                or (item.get("targets")[0] if item.get("targets") else None)
                or item.get("answer")
                or item.get("output")
                or None
            )
            choices = item.get("multiple_choice_targets") or item.get("choices")

        if not inp or label is None:
            log_line("⚠️ Item ohne input/label – skip.")
            continue

        out.append(Example(str(inp), str(label), choices))

    return out

# =========================================================
# ZERO-SHOT Evaluation
# =========================================================

def run_zero_shot_on_dataset(
    prompt: str,
    dataset: List[Example],
    split_name: str,
    model: str = MODEL_NAME,
    system_message: Optional[str] = None,
) -> EvalRunResult:
    client = get_client()
    results: List[Dict[str, Any]] = []
    correct = 0

    for idx, ex in enumerate(dataset):
        log_line(f"[ZERO-SHOT][{split_name}] Processing item {idx+1}/{len(dataset)}")

        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": f"{prompt}\n\n{ex.input}"})

        log_prompt_to_file(messages, "zero-shot", split_name, idx)

        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0
        )

        raw = (resp.choices[0].message.content or "").strip()
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

    acc = correct / len(dataset) if dataset else 0.0
    log_line(f"[ZERO-SHOT][{split_name}] Accuracy = {acc:.4f}")

    return EvalRunResult(
        mode="zero-shot",
        model=model,
        used_prompt=prompt,
        test_results=results,
        accuracy=acc,
    )


# =========================================================
# FEW-SHOT Evaluation
# =========================================================

def build_few_shot_block(train: List[Example], n: int) -> List[Example]:
    return train[:n]


def format_few_shot_block(block: List[Example]) -> str:
    return "\n\n".join(
        f"Example {i}:\nQuestion: {ex.input}\nAnswer: {ex.label}"
        for i, ex in enumerate(block, start=1)
    )


def evaluate_few_shot(
    train_dataset: List[Example],
    test_dataset: List[Example],
    base_prompt: str,
    split_name: str,
    model: str = MODEL_NAME,
    max_train_examples: int = FEW_SHOT_CONTEXT_EXAMPLES,
) -> EvalRunResult:

    client = get_client()

    block = build_few_shot_block(train_dataset, max_train_examples)
    block_text = format_few_shot_block(block)

    system_prompt = (
        base_prompt.rstrip()
        + "\n\nAdditionally I am giving you the following examples to learn from:\n\n"
        + block_text
        + "\n\nLearn from the given examples and answer the question."
    )

    # kompletten Few-Shot-System-Prompt einmal loggen
    DEBUG_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    debug_file = DEBUG_PROMPT_DIR / f"fewshot_system_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    debug_file.write_text(system_prompt, encoding="utf-8")
    log_line(f"System prompt vollständig geloggt in: {debug_file}")

    results: List[Dict[str, Any]] = []
    correct = 0

    for idx, ex in enumerate(test_dataset):
        log_line(f"[FEW-SHOT][{split_name}] Processing item {idx+1}/{len(test_dataset)}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex.input},
        ]

        log_prompt_to_file(messages, "few-shot", split_name, idx)

        resp = chat_with_retry(
            client,
            model=model,
            messages=messages,
            temperature=0.0,
        )
        time.sleep(0.1)

        raw = (resp.choices[0].message.content or "").strip()
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

    acc = correct / len(test_dataset) if test_dataset else 0.0
    log_line(f"[FEW-SHOT][{split_name}] Accuracy = {acc:.4f}")

    return EvalRunResult(
        mode="few-shot",
        model=model,
        base_system_prompt=base_prompt,
        used_prompt=system_prompt,
        few_shot_examples=block,
        test_results=results,
        accuracy=acc,
    )


# =========================================================
# RAG – Azure Retrieval
# =========================================================

def retrieve_azure_context(
    azure_search_index: str,
    question: str,
    top_k: int = RAG_TOP_K,
) -> str:
    """
    Fragt den Azure Search Index ab und baut einen RAG-Kontext bestehend
    aus Q/A-Beispielen (inputs + targets) — wie Few-Shot-Beispiele.
    """
    if not (AZURE_SEARCH_ENDPOINT and azure_search_index and AZURE_SEARCH_API_KEY):
        log_line("⚠️ Azure Search nicht konfiguriert – Kontext bleibt leer.")
        return ""

    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{azure_search_index}/docs/search?api-version=2023-11-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_API_KEY,
    }
    payload = {
        "search": question,
        "top": top_k,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        log_line(f"⚠️ Azure-Suche fehlgeschlagen: {e}")
        return ""

    data = resp.json()
    docs = data.get("value", [])

    examples: List[str] = []
    for d in docs:
        q = d.get("inputs") or d.get("content") or d.get("text") or ""
        targets = d.get("targets") or []
        if isinstance(targets, list) and targets:
            a = targets[0]
        else:
            a = ""

        if not q or not a:
            continue

        block = f"Example:\nQuestion: {q}\nAnswer: {a}"
        examples.append(block)

    return "\n\n".join(examples)


def evaluate_rag(
    dataset: List[Example],
    base_prompt: str,
    azure_search_index: str,
    split_name: str,
    model_name: str = MODEL_NAME,
) -> EvalRunResult:
    """
    RAG-Evaluierung:
      - Kontext aus Azure holen
      - System-Prompt = Base-Prompt + Kontext + Hinweis
      - LLM-Aufruf
    """
    log_header(f"[RAG] Starting RAG Evaluation for split '{split_name}'")

    client = get_client()
    results: List[Dict[str, Any]] = []
    correct = 0

    log_sub("Dataset info")
    log_line(f"Number of examples: {len(dataset)}")

    for idx, ex in enumerate(dataset):
        log_line(f"[RAG][{split_name}] Processing item {idx+1}/{len(dataset)}")

        context = retrieve_azure_context(azure_search_index, ex.input, RAG_TOP_K)

        if context:
            system_message = (
                base_prompt.rstrip()
                + "Additionally I am giving you following examples to learn from:\n\n"
                + context
                + "\n\nLearn from the given examples and answer the following question."
            )
        else:
            system_message = base_prompt.rstrip()

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": ex.input},
        ]

        log_prompt_to_file(messages, "rag", split_name, idx)

        resp = chat_with_retry(
            client,
            model=model_name,
            messages=messages,
            temperature=0.0,
        )
        time.sleep(0.1)

        answer_raw = (resp.choices[0].message.content or "").strip()
        gold_raw = ex.label.strip()

        answer_aligned = align_model_answer(
            answer_raw=answer_raw,
            gold_raw=gold_raw,
            choices=ex.choices,
        )
        answer_norm = normalize_answer(answer_aligned)
        gold_norm = normalize_answer(gold_raw)

        judgement = (answer_norm == gold_norm)
        if judgement:
            correct += 1

        record: Dict[str, Any] = {
            "Index": idx,
            "Input": ex.input,
            "correctAnswer": gold_raw,
            "MbaseAnswer": answer_aligned,
            "MbaseAnswerRaw": answer_raw,
            "judgement": judgement,
            "context": context,
            "system_message": system_message,
        }
        if ex.choices is not None:
            record["choices"] = ex.choices

        results.append(record)

    accuracy = correct / len(dataset) if dataset else 0.0
    log_sub("RAG Evaluation finished")
    log_line(f"[RAG][{split_name}] Accuracy = {accuracy:.4f}")

    return EvalRunResult(
        mode="rag",
        model=model_name,
        base_system_prompt=base_prompt,
        used_prompt=base_prompt,  # Kontext ist per Beispiel unterschiedlich
        few_shot_examples=[],
        test_results=results,
        accuracy=accuracy,
    )


# =========================================================
# SAVE JSON
# =========================================================

def save_run_to_json(
    path: Path,
    *,
    dataset_path: str,
    dataset_size: int,
    config: Dict[str, Any],
    result: EvalRunResult,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "path": dataset_path,
            "num_examples": dataset_size,
        },
        "config": config,
        "result": {
            "mode": result.mode,
            "prompt": result.used_prompt or config.get("prompt", ""),
            "accuracy": result.accuracy,
            "results": result.test_results,
        },
    }

    if result.mode == "few-shot":
        payload["dataset"]["few_shot_examples"] = len(result.few_shot_examples)
        payload["config"]["base_system_prompt"] = result.base_system_prompt

    if result.mode == "rag":
        payload["config"]["base_system_prompt"] = result.base_system_prompt
        payload["config"]["azure_search_index"] = config.get("azure_search_index")

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log_header(f"Saved: {path}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    system_message = None

    dataset_id = (
      HF_DATASET_NAME
      if HF_TASK_NAME in (None, "", "None")
      else f"{HF_DATASET_NAME}_{HF_TASK_NAME}"
    )

    root = RESULTS_ROOT / f"{dataset_id}_{EVAL_MODE}_new"

    # ---------------- ZERO-SHOT ----------------
    if EVAL_MODE == "zero-shot":

        config_base = {
            "prompt": ZERO_SHOT_PROMPT,
            "model_name": MODEL_NAME,
            "mode": "zero-shot",
            "hf_dataset": HF_DATASET_NAME,
            "hf_task_name": HF_TASK_NAME,
        }

        for split_name, max_n in ZERO_SHOT_EVAL_SPLITS.items():
            log_header(f"[ZERO-SHOT] Split '{split_name}'")

            dataset = load_hf_dataset(split_name, max_n)
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

    # ---------------- FEW-SHOT ----------------
    elif EVAL_MODE == "few-shot":

        config_base = {
            "base_system_prompt": FEW_SHOT_BASE_SYSTEM_PROMPT,
            "model_name": MODEL_NAME,
            "mode": "few-shot",
            "hf_dataset": HF_DATASET_NAME,
            "hf_task_name": HF_TASK_NAME,
        }

        log_header(f"[FEW-SHOT] Lade Trainingsdaten '{FEW_SHOT_TRAIN_SPLIT}'")
        train_data = load_hf_dataset(
            FEW_SHOT_TRAIN_SPLIT, FEW_SHOT_TRAIN_MAX
        )
        if not train_data:
            log_line("❌ Keine Trainingsdaten – Few-Shot nicht möglich.")
        else:
            log_line(f"Train examples: {len(train_data)}")

            for split_name, max_n in FEW_SHOT_TEST_SPLITS.items():
                log_header(f"[FEW-SHOT] Test Split '{split_name}'")

                test_data = load_hf_dataset(split_name, max_n)
                if not test_data:
                    log_line("⚠️ Keine Testdaten – skip.")
                    continue

                result = evaluate_few_shot(
                    train_dataset=train_data,
                    test_dataset=test_data,
                    base_prompt=FEW_SHOT_BASE_SYSTEM_PROMPT,
                    split_name=split_name,
                    model=MODEL_NAME,
                    max_train_examples=FEW_SHOT_CONTEXT_EXAMPLES,
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
                        "few_shot_context_examples": FEW_SHOT_CONTEXT_EXAMPLES,
                    },
                    result=result,
                )

    # ---------------- RAG ----------------
    elif EVAL_MODE == "rag":

        config_base = {
            "base_system_prompt": RAG_BASE_SYSTEM_PROMPT,
            "model_name": MODEL_NAME,
            "mode": "rag",
            "hf_dataset": HF_DATASET_NAME,
            "hf_task_name": HF_TASK_NAME,
            "azure_search_index": AZURE_SEARCH_INDEX,
            "rag_top_k": RAG_TOP_K,
        }

        for split_name, max_n in RAG_EVAL_SPLITS.items():
            log_header(f"[RAG] Split '{split_name}'")

            dataset = load_hf_dataset(split_name, max_n)
            if not dataset:
                log_line("⚠️ Keine Daten – skip.")
                continue

            result = evaluate_rag(
                dataset=dataset,
                base_prompt=RAG_BASE_SYSTEM_PROMPT,
                azure_search_index=AZURE_SEARCH_INDEX,
                split_name=split_name,
                model_name=MODEL_NAME,
            )

            out_dir = root / split_name
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = out_dir / f"rag_{HF_TASK_NAME}_{split_name}_{ts}.json"

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

    else:
        raise ValueError("EVAL_MODE muss 'zero-shot', 'few-shot' oder 'rag' sein.")
