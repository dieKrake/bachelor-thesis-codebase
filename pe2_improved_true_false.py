"""
PE2: Prompt Engineering a Prompt Engineer – Python-Implementierung mit Logging.

Diese Version folgt enger dem Algorithmus aus dem Paper:
- Eingabe-Datensatz D wird in T_train, T_dev, T_test aufgeteilt.
- Prompt-Optimierung läuft auf T_train (Fehler-Batches) + T_dev (Auswahl der besten Prompts).
- Am Ende wird der beste Prompt auf T_test evaluiert.
"""

from __future__ import annotations

import random
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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


@dataclass
class PromptCandidate:
    """
    Repräsentiert einen Prompt-Kandidaten.
    accuracy = Dev-Accuracy (wird für Auswahl / Sortierung genutzt).
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


# =========================================================
# Mtask: Prompt auf Datensatz evaluieren
# =========================================================

def evaluate_prompt_on_dataset(
    candidate_prompt: str,
    dataset: List[Example],
    *,
    model: str = "gpt-4o-mini",
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
          "correctAnswer": gold,
          "MtaskAnswer": answer,
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

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )

        answer = (resp.choices[0].message.content or "").strip()
        gold = ex.label.strip()

        judgement = (answer == gold)
        if judgement:
            correct += 1

        eval_results.append(
            {
                "Index": idx,
                "Input": ex.input,
                "correctAnswer": gold,
                "MtaskAnswer": answer,
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
    model: str = "gpt-4o-mini",
    system_message: Optional[str] = None,
) -> PromptCandidate:
    """
    Evaluiert einen Prompt auf T_dev (für Accuracy) und T_train (für Fehler-Beispiele).
    """
    log_sub("Evaluating prompt with Mtask")
    log_line(f"Prompt (first 100 chars): {candidate_prompt[:100]}...")

    # 1) Dev-Set: wird verwendet, um Kandidaten zu vergleichen (Ziel: max. Dev-Accuracy)
    dev_acc, dev_results = evaluate_prompt_on_dataset(
        candidate_prompt,
        dev_dataset,
        model=model,
        system_message=system_message,
    )
    log_line(f"Dev Accuracy = {dev_acc:.4f}")

    # 2) Train-Set: dient dazu, Fehlerbeispiele für Mproposal zu sammeln
    train_acc, train_results = evaluate_prompt_on_dataset(
        candidate_prompt,
        train_dataset,
        model=model,
        system_message=system_message,
    )
    log_line(f"Train Accuracy = {train_acc:.4f}")

    # Log ein paar Beispiele
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
    model: str = "gpt-4o-mini",
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

    resp = client.chat.completions.create(
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
    task_model: str = "gpt-4o-mini",
    proposal_model: str = "gpt-4o-mini",
    iterations: int = 3,
    beam_width: int = 3,            # n: Top-N Prompts pro Schritt
    proposals_per_prompt: int = 2,  # m: neue Prompts pro Kandidat
    failure_batch_size: int = 5,
    system_message_task: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Implementiert eine PE2-ähnliche Beam Search über Prompts,
    orientiert an Algorithmus 1 im Paper.

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
# Dataset laden & splitten
# =========================================================

def load_dataset(path: str) -> List[Example]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset: List[Example] = []

    for item in raw:
        # finde das Input-Feld
        inp = (
            item.get("input")
            or item.get("inputs")
            or item.get("Input")
            or item.get("instruction")
            or item.get("question")
            or None
        )

        # finde das Label-Feld
        label = (
            item.get("label")
            or (item.get("targets")[0] if item.get("targets") else None)
            or item.get("correctAnswer")
            or item.get("answer")
            or item.get("output")
            or None
        )

        if inp is None or label is None:
            print("⚠️ Warnung: Item konnte nicht geladen werden:", item)
            continue

        dataset.append(Example(input=str(inp), label=str(label)))

    return dataset


def split_dataset(
    dataset: List[Example],
    train_size: int,
    dev_size: int,
    seed: int = 0,
) -> Tuple[List[Example], List[Example], List[Example]]:
    """
    Splittet einen Datensatz in T_train, T_dev, T_test.

    Im Paper wird z.B. für MultiArith ein Split von 100/100/Rest genutzt.:contentReference[oaicite:1]{index=1}
    """
    n = len(dataset)
    if train_size + dev_size > n:
        raise ValueError(
            f"train_size + dev_size ({train_size + dev_size}) > dataset size ({n})"
        )

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    train_idx = indices[:train_size]
    dev_idx = indices[train_size : train_size + dev_size]
    test_idx = indices[train_size + dev_size :]

    train = [dataset[i] for i in train_idx]
    dev = [dataset[i] for i in dev_idx]
    test = [dataset[i] for i in test_idx]

    log_sub("Dataset split")
    log_line(f"Total: {n} | Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}")

    return train, dev, test


# =========================================================
# Ergebnis in JSON speichern
# =========================================================

def save_run_to_json(
    output_path: Path,
    *,
    dataset_path: str,
    total_size: int,
    train_size: int,
    dev_size: int,
    test_size: int,
    config: Dict[str, Any],
    result: Dict[str, Any],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "path": dataset_path,
            "num_examples_total": total_size,
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
# Nutzung
# =========================================================

if __name__ == "__main__":
    # Pfad zum Gesamtdatensatz (wird in T_train, T_dev, T_test gesplittet)
    dataset_path = "navigate_full.json"  # <- hier Pfad anpassen

    log_header("Loading full dataset")
    full_dataset = load_dataset(dataset_path)
    log_line(f"Loaded {len(full_dataset)} examples")

    # Split-Parameter (analog zum Paper z.B. 100/100/Rest)
    TRAIN_SIZE = 100
    DEV_SIZE = 100
    train_set, dev_set, test_set = split_dataset(
        full_dataset, train_size=TRAIN_SIZE, dev_size=DEV_SIZE, seed=42
    )

    # Initialer Prompt
    initial_prompt = (
        'You MUST answer with exactly one string: either "True" or "False". '
        'Do not output explanations, punctuation, or additional text. '
        '"True" = you return to the starting point. '
        '"False" = you do not return to the starting point.\n\n'
        'Now answer the following question with "True" or "False":'
    )

    # Meta-Prompt (PE2-artig, hier deine Variante aus dem alten Skript)
    meta_prompt_template = """
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

    # Konfiguration für diesen Run (für JSON)
    iterations = 3
    beam_width = 3
    proposals_per_prompt = 2
    failure_batch_size = 5
    task_model = "gpt-4o-mini"
    proposal_model = "gpt-4o-mini"

    config = {
        "initial_prompt": initial_prompt,
        "meta_prompt_template_snippet": meta_prompt_template[:200],
        "task_model": task_model,
        "proposal_model": proposal_model,
        "iterations": iterations,
        "beam_width": beam_width,
        "proposals_per_prompt": proposals_per_prompt,
        "failure_batch_size": failure_batch_size,
        "train_size": TRAIN_SIZE,
        "dev_size": DEV_SIZE,
        "test_size": len(test_set),
    }

    # PE2-Optimierung auf T_train + T_dev
    result = pe2_optimize_prompt(
        train_dataset=train_set,
        dev_dataset=dev_set,
        initial_prompt=initial_prompt,
        meta_prompt_template=meta_prompt_template,
        task_model=task_model,
        proposal_model=proposal_model,
        iterations=iterations,
        beam_width=beam_width,
        proposals_per_prompt=proposals_per_prompt,
        failure_batch_size=failure_batch_size,
    )

    best_prompt = result["best_prompt"]
    best_dev_acc = result["best_accuracy"]

    # Finaler Testlauf auf T_test mit dem besten Prompt
    log_header("Final evaluation on T_test")
    test_accuracy, test_results = evaluate_prompt_on_dataset(
        best_prompt,
        test_set,
        model=task_model,
        system_message=None,
    )
    log_line(f"Test Accuracy (T_test) = {test_accuracy:.4f}")

    # Für JSON-Ergebnis Test-Infos ergänzen
    final_result = {
        **result,
        "test_accuracy": test_accuracy,
        "test_results": test_results,
    }

    # Konsole: Zusammenfassung
    print("\n\n========= FINAL RESULT =========")
    print("Best Prompt:\n")
    print(best_prompt)
    print("\nBest Dev Accuracy:", best_dev_acc)
    print("Test Accuracy:", test_accuracy)

    print("\nHistory (iteration, dev_accuracy, prompt snippet):")
    for h in final_result["history"]:
        snippet = h["prompt"].replace("\n", " ")[:100]
        print(f"Iter {h['iteration']}: acc_dev={h['dev_accuracy']:.3f} | {snippet}...")

    # JSON speichern
    results_dir = Path("pe2_results")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"pe2_run_{timestamp_str}.json"

    save_run_to_json(
        output_path=output_file,
        dataset_path=dataset_path,
        total_size=len(full_dataset),
        train_size=len(train_set),
        dev_size=len(dev_set),
        test_size=len(test_set),
        config=config,
        result=final_result,
    )
