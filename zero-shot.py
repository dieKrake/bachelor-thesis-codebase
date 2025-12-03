"""
Zero-Shot Evaluation – angelehnt an den PE2-Algorithmus mit Logging.

- Lädt ein gelabeltes Dataset (Input + Label).
- Nutzt einen festen Prompt (Zero-Shot) zur Evaluation auf dem ganzen Datensatz.
- Verwendet dieselbe Mtask-Logik wie im PE2-Skript.
- Speichert Accuracy + Einzelergebnisse als JSON mit Timestamp & Dataset-Info.
"""

from __future__ import annotations

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

# =========================================================
# Logging Helpers (wie im PE2-Skript)
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
# Datentypen (kompatibel zum PE2-Skript)
# =========================================================

@dataclass
class Example:
    input: str
    label: str


@dataclass
class PromptCandidate:
    prompt: str
    accuracy: float = 0.0
    eval_results: List[Dict[str, Any]] = field(default_factory=list)


# =========================================================
# OpenAI-Client
# =========================================================

def get_client() -> OpenAI:
    """
    Erwartet einen OpenAI-Key in der Umgebungsvariable OPENAI_API_KEY,
    oder du trägst ihn hier direkt ein.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Bitte setze OPENAI_API_KEY in der Umgebung "
            "oder trage den Key direkt in get_client() ein."
        )
    return OpenAI(api_key=api_key)


# =========================================================
# Mtask: Prompt auf Datensatz evaluieren
# =========================================================

def run_mtask_on_dataset(
    candidate_prompt: str,
    dataset: List[Example],
    model: str = "gpt-4o-mini",
    system_message: Optional[str] = None,
) -> PromptCandidate:
    """
    Evaluiert einen Prompt auf einem Datensatz.
    Gibt einen PromptCandidate mit Accuracy und detaillierten Ergebnissen zurück.
    """
    log_sub("Evaluating prompt with Mtask (Zero-Shot)")
    log_line(f"Prompt (first 100 chars): {candidate_prompt[:100]}...")

    client = get_client()
    eval_results: List[Dict[str, Any]] = []
    correct = 0

    for idx, ex in enumerate(dataset):
        # User-Query: Prompt + Input
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
                "MbaseAnswer": answer,
                "judgement": judgement,
            }
        )

        log_line(f"Example {idx}: model={answer}, gold={gold}, ok={judgement}")

    accuracy = correct / len(dataset) if dataset else 0.0
    log_line(f"Accuracy = {accuracy:.4f}")

    return PromptCandidate(
        prompt=candidate_prompt,
        accuracy=accuracy,
        eval_results=eval_results,
    )


# =========================================================
# Dataset laden
# =========================================================

def load_dataset(path: str) -> list[Example]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset: List[Example] = []

    for item in raw:
        # Input-Feld finden
        inp = (
            item.get("input")
            or item.get("inputs")
            or item.get("Input")
            or item.get("instruction")
            or None
        )

        # Label-Feld finden
        label = (
            item.get("label")
            or (item.get("targets")[0] if "targets" in item and item.get("targets") else None)
            or item.get("correctAnswer")
            or item.get("output")
            or None
        )

        if inp is None or label is None:
            print("⚠️ Warnung: Item konnte nicht geladen werden:", item)
            continue

        dataset.append(Example(input=str(inp), label=str(label)))

    return dataset


# =========================================================
# Zero-Shot Evaluation Wrapper
# =========================================================

def zero_shot_evaluate(
    dataset: List[Example],
    prompt: str,
    model_name: str = "gpt-4o-mini",
    system_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Führt eine einfache Zero-Shot-Evaluation durch:
    Ein Prompt, einmal über den ganzen Datensatz, kein Prompt-Update.
    Nutzt intern run_mtask_on_dataset (wie PE2 Mtask).
    """
    candidate = run_mtask_on_dataset(
        candidate_prompt=prompt,
        dataset=dataset,
        model=model_name,
        system_message=system_message,
    )

    return {
        "prompt": candidate.prompt,
        "accuracy": candidate.accuracy,
        "results": candidate.eval_results,
    }


# =========================================================
# Ergebnis in JSON speichern (an PE2 angelehnt)
# =========================================================

def save_run_to_json(
    output_path: Path,
    *,
    dataset_path: str,
    dataset_size: int,
    config: Dict[str, Any],
    result: Dict[str, Any],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "path": dataset_path,
            "num_examples": dataset_size,
        },
        "config": config,
        "result": result,
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_header(f"Saved Zero-Shot run to JSON: {output_path}")


# =========================================================
# Nutzung / Main
# =========================================================

if __name__ == "__main__":
    # Pfad zum Dataset – analog zum PE2-Skript
    dataset_path = "navigate_200/dataset_random/navigate_40_test.json"  # <- bei Bedarf anpassen

    log_header("Loading dataset")
    dataset = load_dataset(dataset_path)
    log_line(f"Loaded {len(dataset)} examples")

    # Zero-Shot Prompt
    zero_shot_prompt = (
        'You MUST answer with exactly one string: either "True" or "False". '
        'Do not output explanations, punctuation, or additional text. '
        '"True" = you return to the starting point. '
        '"False" = you do not return to the starting point.\n\n'
        'Now answer the following question with "True" or "False":'
    )

    pe2_prompt = ("")

    # Optional: separates system_message
    system_message_task = None

    # Konfiguration wie bei PE2, nur schlanker
    model_name = "gpt-4o-mini"

    config = {
        "prompt": zero_shot_prompt,
        "model_name": model_name,
        "system_message_snippet": (system_message_task or "")[:200],
        "mode": "zero-shot",
    }

    # Zero-Shot Evaluation ausführen
    log_header("Running Zero-Shot Evaluation")
    result = zero_shot_evaluate(
        dataset=dataset,
        prompt=zero_shot_prompt,
        model_name=model_name,
        system_message=system_message_task,
    )

    # Konsole: Zusammenfassung
    print("\n\n========= ZERO-SHOT RESULT =========")
    print("Prompt:\n")
    print(result["prompt"])
    print("\nAccuracy:", result["accuracy"])

    print("\nExamples (first 5):")
    for rec in result["results"][:5]:
        print(
            f"[{rec['Index']}] acc={rec['judgement']} | gold={rec['correctAnswer']} | model={rec['MbaseAnswer']}"
        )

    # JSON speichern (analog zu pe2_results/…)
    results_dir = Path("navigate_200/zero_shot_results")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"zeroshot_eval_{timestamp_str}.json"

    save_run_to_json(
        output_path=output_file,
        dataset_path=dataset_path,
        dataset_size=len(dataset),
        config=config,
        result=result,
    )
