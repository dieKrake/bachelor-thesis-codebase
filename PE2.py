"""
PE2: Prompt Engineering a Prompt Engineer – HF-Version mit BIG-Bench- und BoolQ-Support

- Lädt einen HF-Datensatz (z. B. tasksource/bigbench, logical_fallacy_detection oder google/boolq).
- Verwendet:
    * T_train (Fehlerbeispiele für Mproposal)
    * T_dev   (Validation / Dev-Accuracy)
    * T_test  (optionale finale Evaluation)

- Prompt-Optimierung läuft wie im Paper:
    * Mtask: evaluiert Prompt-Kandidaten auf T_train/T_dev
    * Mproposal: erzeugt neue Prompts aus Fehlbeispielen
    * Beam-artige Suche über mehrere Iterationen
"""

from __future__ import annotations

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from datasets import load_dataset as hf_load_dataset
from openai import OpenAI, RateLimitError
import time
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 🔧 KONFIGURATION
# =========================================================

# HuggingFace Dataset-Konfiguration
# Für BoolQ:
# HF_DATASET_NAME = "google/boolq"
# HF_TASK_NAME = None
#
# Für BIG-Bench / andere:
# HF_DATASET_NAME = "tasksource/bigbench"
# HF_TASK_NAME = "navigate"
HF_DATASET_NAME = "tasksource/bigbench"          # z.B. "tasksource/bigbench" oder "google/boolq"
HF_TASK_NAME: Optional[str] = "fact_checker"      # z.B. "navigate" oder None für BoolQ

HF_TRAIN_SPLIT = "train"
HF_DEV_SPLIT = "dev"
HF_TEST_SPLIT: Optional[str] = "validation"   # für BoolQ: validation; sonst z. B. "test" oder None

# Für BoolQ:
#   Train: erste 100 aus train
#   Dev:   nächste 200 aus train
#   Test:  erste 200 aus validation
TRAIN_MAX = 50          # max. Anzahl Beispiele für T_train
DEV_MAX = 50         # max. Anzahl Beispiele für T_dev
TEST_MAX: Optional[int] = 200  # max. Beispiele aus HF_TEST_SPLIT (für BoolQ: 200)

RANDOM_SEED = 42

# Model-Konfiguration
TASK_MODEL = "gpt-4o-mini"
PROPOSAL_MODEL = "gpt-4o-mini"

# Initialer Prompt – z.B. wie Navigate/BoolQ-True/False
INITIAL_PROMPT = (
  'You MUST answer with exactly one string: either "true" or "false". '
  'Do not output explanations, reasoning steps, punctuation, or additional text. '
  '"true" = the claimed fact is true. '
  '"false" = the claimed fact is false.\n\n'
  'Now answer the following question with exactly one word: "true" or "false".'
)

# Meta-Prompt
META_PROMPT_TEMPLATE = """
We are refining a prompt for a classification task.

# Current Prompt
{prompt}

# Failed Examples
The following examples are cases where the model failed (wrong judgement):

{examples}

# Instructions
1. Analyze why the current prompt might have led to these mistakes.
2. Propose an improved version of the prompt that:
  - stays concise,
  - keeps the same task and output format,
  - reduces the likelihood of these failures.

Return your answer as valid JSON with the following schema:

{{
  "reasoning": "short explanation of the changes",
  "improved_prompt": "the new improved prompt"
}}
""".strip()

# PE2-Hyperparameter
ITERATIONS = 3
BEAM_WIDTH = 2
PROPOSALS_PER_PROMPT = 2
FAILURE_BATCH_SIZE = 5

# Ergebnis-Ordner
RESULTS_ROOT = Path("pe2_hf_results")


# =========================================================
# Logging Helpers
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
# Datentypen
# =========================================================

@dataclass
class Example:
    input: str
    label: str
    choices: Optional[List[str]] = None  # für Multiple-Choice-Datensätze


@dataclass
class PromptCandidate:
    """
    Repräsentiert einen Prompt-Kandidaten.
    accuracy = Dev-Accuracy (für Auswahl / Sortierung).
    """
    prompt: str
    accuracy: float = 0.0  # Dev-Accuracy
    eval_results_dev: List[Dict[str, Any]] = field(default_factory=list)
    eval_results_train: List[Dict[str, Any]] = field(default_factory=list)


# =========================================================
# OpenAI-Client
# =========================================================

def get_client() -> OpenAI:
    """
    Erwartet OPENAI_API_KEY in der Umgebung.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Bitte OPENAI_API_KEY als Umgebungsvariable setzen.")
    return OpenAI(api_key=api_key)

def chat_with_retry(client: OpenAI, **kwargs):
    """
    Wrapper um client.chat.completions.create mit einfachem Retry bei Rate-Limit.
    - Bei 'per minute/second' Limits: exponentielles Backoff und nochmal probieren.
    - Bei 'requests per day' Limit: sauber abbrechen.
    """
    max_retries = 5
    base_delay = 1.0  # Sekunden

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            msg = str(e)
            # Tageslimit: bringt nichts zu warten -> direkt abbrechen
            if "requests per day" in msg or "RPD" in msg:
                log_line("❌ Daily rate limit reached – aborting PE2 run cleanly.")
                raise

            # Sonst: klassisches Backoff (pro Minute/sek)
            wait = base_delay * (2 ** attempt)
            log_line(
                f"⏳ Rate limit (attempt {attempt+1}/{max_retries}), "
                f"waiting {wait:.1f}s before retry..."
            )
            time.sleep(wait)

    # Wenn nach max_retries immer noch RateLimitError:
    raise RuntimeError("Rate limit persisted after retries – aborting.")

# =========================================================
# Normalisierung & Alignment (analog HF-Eval-Skript)
# =========================================================

def normalize_answer(text: str) -> str:
    """
    Normalisiert Modellantworten und Labels, um verschiedene Schreibweisen
    robuster vergleichen zu können (True/False, Yes/No, choice:42, ...).
    """
    if text is None:
        return ""

    t = str(text).strip()

    # einfache Endzeichen weg (z.B. "True." -> "True")
    if t.endswith((".", "!", "?")):
        t = t[:-1]

    t = t.strip().lower()

    # "choice:50" -> "50"
    if t.startswith("choice:"):
        t = t[len("choice:"):].strip()

    # Booleans vereinheitlichen
    bool_map = {
        "true": "bool_true",
        "yes": "bool_true",
        "y": "bool_true",
        "false": "bool_false",
        "no": "bool_false",
        "n": "bool_false",
    }
    if t in bool_map:
        return bool_map[t]

    return t


def align_model_answer(
    answer_raw: str,
    gold_raw: str,
    choices: Optional[List[str]] = None,
) -> str:
    """
    Versucht, aus der rohen Modellantwort (answer_raw) die eigentliche
    Antwort zu extrahieren, ohne falsche Substring-Matches zu erzeugen.
    """
    if answer_raw is None:
        return ""

    t = answer_raw.strip()

    # 1) "choice:50" -> "50"
    lower = t.lower()
    if lower.startswith("choice:"):
        t = t.split(":", 1)[1].strip()
        lower = t.lower()

    # 2) exakte Übereinstimmung mit Gold-Label
    if gold_raw is not None:
        gold_str = str(gold_raw).strip()
        if t == gold_str:
            return gold_str

    # 3) exakte Übereinstimmung mit einer der Choices
    if choices:
        norm_choices = [str(c).strip() for c in choices]

        # 3a) exakte Gleichheit
        for c in norm_choices:
            if t == c:
                return c

        # 3b) Token-basiert
        tokens = t.replace(",", " ").replace(".", " ").split()
        for tok in tokens:
            for c in norm_choices:
                if tok == c:
                    return c

    # 4) Fallback: bereinigte Rohantwort zurückgeben
    return t


# =========================================================
# Mtask: Prompt auf Datensatz evaluieren (Dev oder Train)
# =========================================================

def evaluate_prompt_on_dataset(
    candidate_prompt: str,
    dataset: List[Example],
    *,
    model: str = TASK_MODEL,
    system_message: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluiert einen Prompt auf einem Datensatz und liefert (accuracy, eval_results).

    eval_results:
      [
        {
          "Index": idx,
          "Input": ex.input,
          "correctAnswer": gold_raw,
          "MtaskAnswer": answer_aligned,
          "MtaskAnswerRaw": answer_raw,
          "judgement": bool,
        }, ...
      ]
    """
    client = get_client()
    eval_results: List[Dict[str, Any]] = []
    correct = 0

    if max_examples is not None:
        dataset_iter = dataset[:max_examples]
    else:
        dataset_iter = dataset

    for idx, ex in enumerate(dataset_iter):
        user_content = f"{candidate_prompt}\n\n{ex.input}"

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_content})

        resp = chat_with_retry(
            client,
            model=model,
            messages=messages,
            temperature=0.0,
        )

        answer_raw = (resp.choices[0].message.content or "").strip()
        gold_raw = ex.label.strip()

        # HF-kompatible Auswertung
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

        eval_results.append(
            {
                "Index": idx,
                "Input": ex.input,
                "correctAnswer": gold_raw,
                "MtaskAnswer": answer_aligned,
                "MtaskAnswerRaw": answer_raw,
                "judgement": judgement,
            }
        )

    accuracy = correct / len(dataset_iter) if dataset_iter else 0.0
    return accuracy, eval_results


def run_mtask_for_candidate(
    candidate_prompt: str,
    train_dataset: List[Example],
    dev_dataset: List[Example],
    *,
    model: str = TASK_MODEL,
    system_message: Optional[str] = None,
) -> PromptCandidate:
    """
    Evaluiert einen Prompt auf T_dev (für Accuracy) und T_train (für Fehler-Beispiele).
    """
    log_sub("Evaluating prompt with Mtask")
    log_line(f"Prompt (first 100 chars): {candidate_prompt[:100]}...")

    # 1) Dev-Set: für Vergleich der Kandidaten (Dev-Accuracy)
    dev_acc, dev_results = evaluate_prompt_on_dataset(
        candidate_prompt,
        dev_dataset,
        model=model,
        system_message=system_message,
    )
    log_line(f"Dev Accuracy = {dev_acc:.4f}")

    # 2) Train-Set: für Fehlbeispiele (Mproposal)
    train_acc, train_results = evaluate_prompt_on_dataset(
        candidate_prompt,
        train_dataset,
        model=model,
        system_message=system_message,
    )
    log_line(f"Train Accuracy = {train_acc:.4f}")

    # Log ein paar Train-Beispiele
    for r in train_results[:5]:
        log_line(
            f"[Train] Example {r['Index']}: model={r['MtaskAnswer']}, "
            f"gold={r['correctAnswer']}, ok={r['judgement']}"
        )

    return PromptCandidate(
        prompt=candidate_prompt,
        accuracy=dev_acc,
        eval_results_dev=dev_results,
        eval_results_train=train_results,
    )


# =========================================================
# Hilfsfunktion: Fehlbeispiele formatieren (Batch B)
# =========================================================

def build_failure_examples_text(
    eval_results: List[Dict[str, Any]],
    failure_batch_size: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Sammelt Fehlbeispiele aus eval_results (typischerweise T_train),
    sampelt ggf. und baut einen Textblock B für den Meta-Prompt.
    """
    failures = [r for r in eval_results if not r.get("judgement", False)]
    log_sub("Collecting failure examples")
    log_line(f"Found {len(failures)} failures")

    if not failures:
        return "", []

    if len(failures) > failure_batch_size:
        failures = random.sample(failures, failure_batch_size)
        log_line(f"Sampled down to {failure_batch_size} failures")

    blocks = []
    for f in failures:
        block = (
            "Input:\n"
            + f["Input"]
            + "\n\nLLM Answer: "
            + str(f["MtaskAnswer"])
            + "\nCorrect Answer: "
            + str(f["correctAnswer"])
            + "\nJudgement: "
            + str(f["judgement"])
        )
        blocks.append(block)

    examples_text = "\n\n---\n\n".join(blocks)
    return examples_text, failures


# =========================================================
# Mproposal: Meta-Prompt, der verbesserte Prompts erzeugt
# =========================================================

def run_mproposal(
    current_prompt: str,
    train_eval_results: List[Dict[str, Any]],
    meta_prompt_template: str,
    failure_batch_size: int,
    model: str = PROPOSAL_MODEL,
) -> Optional[str]:
    """
    Erzeugt genau EINEN verbesserten Prompt aus einem Meta-Prompt-Aufruf.
    Erwartet, dass das Modell ein JSON mit Key "improved_prompt" ausgibt.
    """
    log_sub("Running Mproposal")

    client = get_client()

    examples_text, failures = build_failure_examples_text(
        train_eval_results, failure_batch_size
    )
    if not failures:
        log_line("No failures → no improved prompt needed")
        return None

    meta_prompt = meta_prompt_template.format(
        prompt=current_prompt,
        examples=examples_text,
    )

    log_line("Sending Meta-Prompt to proposal model")

    resp = chat_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": meta_prompt},
        ],
        temperature=0.0,
      )

    raw = (resp.choices[0].message.content or "").strip()
    log_line(f"Proposal raw response (first 200 chars): {raw[:200]}...")

    try:
        parsed = json.loads(raw)
        improved_prompt = parsed.get("improved_prompt")
        if isinstance(improved_prompt, str) and improved_prompt.strip():
            log_line("Improved prompt extracted successfully")
            return improved_prompt.strip()
        else:
            log_line("JSON parsed but no 'improved_prompt' key found")
    except Exception:
        log_line("Failed to parse JSON from proposal response")

    return None


# =========================================================
# PE2-ähnlicher Beam-Algorithmus (mit T_train / T_dev)
# =========================================================

def pe2_optimize_prompt(
    train_dataset: List[Example],
    dev_dataset: List[Example],
    initial_prompt: str,
    meta_prompt_template: str,
    task_model: str = TASK_MODEL,
    proposal_model: str = PROPOSAL_MODEL,
    iterations: int = ITERATIONS,
    beam_width: int = BEAM_WIDTH,
    proposals_per_prompt: int = PROPOSALS_PER_PROMPT,
    failure_batch_size: int = FAILURE_BATCH_SIZE,
    system_message_task: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Implementiert eine PE2-ähnliche Beam Search über Prompts.

    - Promptqualität wird über Dev-Accuracy (T_dev) gemessen.
    - Fehlerbeispiele für Mproposal kommen aus T_train.
    """
    log_header("Starting PE2 Optimization")

    # Start mit einem Prompt (Iteration 0)
    current_candidates: List[PromptCandidate] = []

    base_candidate = run_mtask_for_candidate(
        candidate_prompt=initial_prompt,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        model=task_model,
        system_message=system_message_task,
    )
    current_candidates.append(base_candidate)

    best_prompt = base_candidate.prompt
    best_accuracy = base_candidate.accuracy

    history: List[Dict[str, Any]] = [
        {
            "iteration": 0,
            "prompt": base_candidate.prompt,
            "dev_accuracy": base_candidate.accuracy,
        }
    ]

    # Alle bisher erzeugten Kandidaten (für Backtracking)
    all_candidates: List[PromptCandidate] = [base_candidate]

    for it in range(1, iterations + 1):
        log_header(f"ITERATION {it}")

        # Besten Beam über ALLE bisherigen Kandidaten wählen (Backtracking)
        all_candidates.sort(key=lambda c: c.accuracy, reverse=True)
        parents = all_candidates[:beam_width]

        log_line("Top prompts entering this iteration:")
        for p in parents:
            log_line(f"  acc_dev={p.accuracy:.4f} | {p.prompt[:80]}...")

        # Early Stop, falls bereits perfekte Dev-Accuracy
        if parents and parents[0].accuracy >= 1.0:
            log_line("Perfect dev accuracy reached → stopping early")
            break

        new_candidates: List[PromptCandidate] = []

        # Für jeden Parent mehrere Vorschläge holen
        for parent in parents:
            for _ in range(proposals_per_prompt):
                log_sub("Generating improved prompt proposal")
                improved = run_mproposal(
                    current_prompt=parent.prompt,
                    train_eval_results=parent.eval_results_train,
                    meta_prompt_template=meta_prompt_template,
                    failure_batch_size=failure_batch_size,
                    model=proposal_model,
                )

                if not improved:
                    log_line("No valid improved prompt returned")
                    continue

                log_line(f"Received improved prompt: {improved[:80]}...")

                # Duplikate vermeiden (über alle bisherigen + neue Kandidaten)
                if any(c.prompt == improved for c in all_candidates + new_candidates):
                    log_line("Improved prompt is duplicate → skipping")
                    continue

                # neuen Prompt auf Train+Dev evaluieren
                cand = run_mtask_for_candidate(
                    candidate_prompt=improved,
                    train_dataset=train_dataset,
                    dev_dataset=dev_dataset,
                    model=task_model,
                    system_message=system_message_task,
                )
                new_candidates.append(cand)
                all_candidates.append(cand)

                history.append(
                    {
                        "iteration": it,
                        "prompt": cand.prompt,
                        "dev_accuracy": cand.accuracy,
                    }
                )

                # global best updaten (auf Basis Dev-Accuracy)
                if cand.accuracy > best_accuracy:
                    best_accuracy = cand.accuracy
                    best_prompt = cand.prompt

        if not new_candidates:
            log_line("No new candidates generated → stopping.")
            break

        log_sub("End of iteration beam summary (top current candidates by dev accuracy)")
        all_candidates.sort(key=lambda c: c.accuracy, reverse=True)
        for c in all_candidates[:beam_width]:
            log_line(f"Keep acc_dev={c.accuracy:.4f} | {c.prompt[:80]}...")

    log_header("PE2 Finished")
    log_line(f"Best Prompt (first 120 chars): {best_prompt[:120]}...")
    log_line(f"Best Dev Accuracy: {best_accuracy:.4f}")

    return {
        "best_prompt": best_prompt,
        "best_accuracy": best_accuracy,  # Dev-Accuracy
        "history": history,
        "all_candidates": [
            {
                "prompt": c.prompt,
                "dev_accuracy": c.accuracy,
            }
            for c in all_candidates
        ],
    }


# =========================================================
# HF-Dataset laden (mit BoolQ-Spezialfall)
# =========================================================

def load_hf_dataset(
    split: str,
    max_examples: Optional[int] = None,
) -> List[Example]:
    """
    Lädt ein HF-Dataset (z. B. BIG-Bench-Task oder google/boolq)
    und mappt ihn auf Example-Struktur.
    """

    log_sub(f"Lade HF-Dataset: {HF_DATASET_NAME}"
            + (f"/{HF_TASK_NAME}" if HF_TASK_NAME else "")
            + f"[{split}]")

    # Mit/ohne Task-Name unterscheiden
    if HF_TASK_NAME:
        ds = hf_load_dataset(HF_DATASET_NAME, HF_TASK_NAME, split=split)
    else:
        ds = hf_load_dataset(HF_DATASET_NAME, split=split)

    available = len(ds)
    if max_examples is not None:
        if max_examples > available:
            log_line(
                f"⚠️ Gewünschte Größe {max_examples} > verfügbare Beispiele {available} "
                f"in split '{split}' – verwende {available} Beispiele."
            )
            max_examples = available
        else:
            log_line(f"Verwende {max_examples} von {available} verfügbaren Beispielen.")
        ds = ds.select(range(max_examples))
    else:
        max_examples = available
        log_line(f"Verwende alle {available} Beispiele aus split '{split}'.")

    dataset: List[Example] = []

    for item in ds:
        # Spezieller Pfad für BoolQ – identisch zur Eval-Logik
        if HF_DATASET_NAME == "google/boolq" and HF_TASK_NAME in (None, "", "None"):
            # BoolQ: question (str), passage (str), answer (bool)
            question = item["question"]
            passage = item["passage"]
            answer = item["answer"]

            inp = f"Question: {question}\nPassage: {passage}"
            label = "True" if bool(answer) else "False"
            choices = None

        # Generische Logik für andere Datasets (BIG-Bench etc.)
        else:
            inp = (
                item.get("input")
                or item.get("inputs")
                or item.get("Input")
                or item.get("instruction")
                or item.get("question")
                or None
            )

            label = (
                item.get("label")
                or (
                    item.get("targets")[0]
                    if isinstance(item.get("targets"), list) and item.get("targets")
                    else None
                )
                or item.get("correctAnswer")
                or item.get("answer")
                or item.get("output")
                or None
            )

            choices = None
            mct = item.get("multiple_choice_targets") or item.get("choices")
            if isinstance(mct, list) and mct:
                choices = [str(c) for c in mct]

        if inp is None or label is None:
            print("⚠️ Warnung: HF-Item konnte nicht geladen werden:", item)
            continue

        dataset.append(
            Example(
                input=str(inp),
                label=str(label),
                choices=choices,
            )
        )

    log_line(f"Loaded {len(dataset)} examples from split '{split}'")
    return dataset


# =========================================================
# Ergebnis in JSON speichern
# =========================================================

def save_run_to_json(
    output_path: Path,
    *,
    hf_dataset_name: str,
    hf_task_name: Optional[str],
    train_size: int,
    dev_size: int,
    test_size: int,
    train_split: str,
    dev_split: str,
    test_split: Optional[str],
    config: Dict[str, Any],
    result: Dict[str, Any],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "hf_dataset": hf_dataset_name,
            "hf_task_name": hf_task_name,
            "train_split": train_split,
            "dev_split": dev_split,
            "test_split": test_split,
            "train_size": train_size,
            "dev_size": dev_size,
            "test_size": test_size,
        },
        "config": config,
        "result": result,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_header(f"Saved run to JSON: {output_path}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    log_header("Loading HF splits for PE2 (train/dev[/test])")

    # Sonderlogik für BoolQ: Train/Dev/Test so wählen, dass sie zum HF-Eval-Skript passen
    if HF_DATASET_NAME == "google/boolq" and HF_TASK_NAME in (None, "", "None"):
        # 1) Train für Train+Dev gemeinsam laden
        full_train = load_hf_dataset("train", TRAIN_MAX + DEV_MAX)

        # T_train: exakt dieselben 100 Beispiele wie im HF-Eval-Few-Shot (train[0:100])
        train_set = full_train[:TRAIN_MAX]

        # T_dev: nächste DEV_MAX Beispiele, disjunkt zu T_train (train[100:300])
        dev_set = full_train[TRAIN_MAX:TRAIN_MAX + DEV_MAX]

        # T_test: dieselben 200 Validation-Beispiele wie im HF-Eval-Skript (validation[0:200])
        test_set = load_hf_dataset("validation", TEST_MAX)

        train_split_name = "train[0:100]"
        dev_split_name = "train[100:300]"
        test_split_name = "validation[0:200]"

    else:
        full_train = load_hf_dataset("train", TRAIN_MAX + DEV_MAX)
        train_set = full_train[:TRAIN_MAX]
        dev_set = full_train[TRAIN_MAX:TRAIN_MAX + DEV_MAX]

        if HF_TEST_SPLIT is not None and TEST_MAX is not None:
            test_set = load_hf_dataset(HF_TEST_SPLIT, TEST_MAX)
        else:
            test_set = []

        train_split_name = HF_TRAIN_SPLIT
        dev_split_name = HF_DEV_SPLIT
        test_split_name = HF_TEST_SPLIT if test_set else None

    # Konfiguration für diesen Run (für JSON)
    config = {
        "initial_prompt": INITIAL_PROMPT,
        "meta_prompt_template_snippet": META_PROMPT_TEMPLATE[:400],
        "task_model": TASK_MODEL,
        "proposal_model": PROPOSAL_MODEL,
        "iterations": ITERATIONS,
        "beam_width": BEAM_WIDTH,
        "proposals_per_prompt": PROPOSALS_PER_PROMPT,
        "failure_batch_size": FAILURE_BATCH_SIZE,
        "train_max": TRAIN_MAX,
        "dev_max": DEV_MAX,
        "test_max": TEST_MAX,
    }

    # PE2-Optimierung auf T_train + T_dev
    result = pe2_optimize_prompt(
        train_dataset=train_set,
        dev_dataset=dev_set,
        initial_prompt=INITIAL_PROMPT,
        meta_prompt_template=META_PROMPT_TEMPLATE,
        task_model=TASK_MODEL,
        proposal_model=PROPOSAL_MODEL,
        iterations=ITERATIONS,
        beam_width=BEAM_WIDTH,
        proposals_per_prompt=PROPOSALS_PER_PROMPT,
        failure_batch_size=FAILURE_BATCH_SIZE,
    )

    best_prompt = result["best_prompt"]
    best_dev_acc = result["best_accuracy"]

    # Optionaler finaler Testlauf auf T_test
    if test_set:
        log_header("Final evaluation on T_test (HF test/validation split)")
        test_accuracy, test_results = evaluate_prompt_on_dataset(
            best_prompt,
            test_set,
            model=TASK_MODEL,
            system_message=None,
        )
        log_line(f"Test Accuracy (T_test) = {test_accuracy:.4f}")

        result["test_accuracy"] = test_accuracy
        result["test_results"] = test_results
    else:
        log_header("No HF_TEST_SPLIT configured or test_set empty – skipping final T_test evaluation")
        test_accuracy = None

    # Konsole: Zusammenfassung
    print("\n\n========= FINAL RESULT =========")
    print("Best Prompt:\n")
    print(best_prompt)
    print("\nBest Dev Accuracy:", best_dev_acc)
    if test_set:
        print("Test Accuracy:", test_accuracy)

    print("\nHistory (iteration, dev_accuracy, prompt snippet):")
    for h in result["history"]:
        snippet = h["prompt"].replace("\n", " ")[:100]
        print(f"Iter {h['iteration']}: acc_dev={h['dev_accuracy']:.3f} | {snippet}...")

    # JSON speichern
    results_dir = RESULTS_ROOT
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = HF_TASK_NAME if HF_TASK_NAME else HF_DATASET_NAME.split("/")[-1]
    output_file = results_dir / f"pe2_hf_{task_id}_{timestamp_str}.json"

    save_run_to_json(
        output_path=output_file,
        hf_dataset_name=HF_DATASET_NAME,
        hf_task_name=HF_TASK_NAME,
        train_size=len(train_set),
        dev_size=len(dev_set),
        test_size=len(test_set),
        train_split=train_split_name,
        dev_split=dev_split_name,
        test_split=test_split_name,
        config=config,
        result=result,
    )
