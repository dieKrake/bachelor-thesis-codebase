"""
Few-Shot-Evaluierung eines Prompts auf einem Klassifikations-Datensatz.

- Train-Daten: werden in Few-Shot-Beispiele (Q/A-Paare) umgewandelt.
- Test-Daten: werden mit Base-Prompt + Few-Shot-Beispielen evaluiert.
- Pro Test-Item eine neue, unabhängige LLM-Anfrage.
- Ergebnisse + Metadaten werden als JSON gespeichert.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

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
class EvalRunResult:
    mode: str  # "few-shot"
    model: str
    base_system_prompt: str
    used_prompt: str  # der tatsächlich verwendete System-Prompt (inkl. Few-Shot-Beispiele)
    few_shot_examples: List[Example] = field(default_factory=list)
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    accuracy: float = 0.0


# =========================================================
# OpenAI-Client
# =========================================================

def get_client() -> OpenAI:
    # ❗ Hier deinen API-Key eintragen ODER über Umgebungsvariable OPENAI_API_KEY arbeiten.
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


# =========================================================
# Dataset laden
# =========================================================

def load_dataset(path: str) -> List[Example]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset: List[Example] = []

    for item in raw:
        # Input-Feld suchen
        inp = (
            item.get("input")
            or item.get("inputs")
            or item.get("Input")
            or item.get("question")
            or item.get("instruction")
            or None
        )

        # Label-Feld suchen
        label = (
            item.get("label")
            or item.get("answer")
            or item.get("output")
            or (
                item.get("targets")[0]
                if isinstance(item.get("targets"), list) and item["targets"]
                else None
            )
            or None
        )

        if inp is None or label is None:
            print("⚠️ Warnung: Item konnte nicht geladen werden:", item)
            continue

        dataset.append(Example(input=str(inp), label=str(label)))

    return dataset


# =========================================================
# Few-Shot-Kontext bauen
# =========================================================

def build_few_shot_block(
    train_dataset: List[Example],
    max_examples: int = 8,
) -> List[Example]:
    """
    Wählt bis zu max_examples Beispiele aus dem Train-Set aus (zufälliges Sampling).
    """
    if not train_dataset:
        return []

    if len(train_dataset) <= max_examples:
        return list(train_dataset)

    return random.sample(train_dataset, max_examples)


def format_few_shot_block(examples: List[Example]) -> str:
    """
    Formatiert Few-Shot-Beispiele als Textblock:

    Example 1:
    Question: ...
    Answer: ...

    ...
    """
    lines = []
    for i, ex in enumerate(examples, start=1):
        block = (
            f"Example {i}:\n"
            f"Question: {ex.input}\n"
            f"Answer: {ex.label}"
        )
        lines.append(block)
    return "\n\n".join(lines)


# =========================================================
# Few-Shot-Evaluierung
# =========================================================

def evaluate_few_shot(
    train_dataset: List[Example],
    test_dataset: List[Example],
    base_system_prompt: str,
    model: str = "gpt-4o-mini",
    max_train_examples: int = 40,
) -> EvalRunResult:
    """
    Führt eine Few-Shot-Evaluierung durch:

    - nimmt max_train_examples Beispiele aus dem Train-Set
    - baut daraus einen System-Prompt (Base-Prompt + Q/A-Beispiele)
    - evaluiert alle Test-Beispiele mit diesem System-Prompt
    """
    log_header("Starting Few-Shot Evaluation")

    client = get_client()

    # Few-Shot-Beispiele wählen und formatieren
    log_sub("Selecting few-shot examples")
    few_shot_examples = build_few_shot_block(
        train_dataset, max_examples=max_train_examples
    )
    log_line(f"Using {len(few_shot_examples)} train examples as few-shot context")

    few_shot_text = format_few_shot_block(few_shot_examples)

    # System-Prompt aufbauen (finaler Prompt, der auch im JSON gespeichert wird)
    system_message = (
        base_system_prompt.strip()
        + few_shot_text
        + '\n\nLearn from the given examples and answer the following question with "True" or "False":'
    )

    log_sub("System prompt preview")
    log_line(
        system_message[:300].replace("\n", " ")
        + ("..." if len(system_message) > 300 else "")
    )

    eval_results: List[Dict[str, Any]] = []
    correct = 0

    log_sub("Evaluating test set")
    log_line(f"Number of test examples: {len(test_dataset)}")

    for idx, ex in enumerate(test_dataset):
        # Jede Anfrage ist unabhängig: nur System-Message + aktuelle Frage
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": ex.input},
        ]

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )

        answer = resp.choices[0].message.content.strip()
        gold = ex.label.strip()

        judgement = (answer == gold)
        if judgement:
            correct += 1

        # 👇 Schlüsselname hier an Zero-Shot-Schema angepasst: "MbaseAnswer"
        eval_results.append(
            {
                "Index": idx,
                "Input": ex.input,
                "correctAnswer": gold,
                "MbaseAnswer": answer,
                "judgement": judgement,
            }
        )

        log_line(f"Test {idx}: model={answer}, gold={gold}, ok={judgement}")

    accuracy = correct / len(test_dataset) if test_dataset else 0.0
    log_sub("Evaluation finished")
    log_line(f"Accuracy = {accuracy:.4f}")

    return EvalRunResult(
        mode="few-shot",
        model=model,
        base_system_prompt=base_system_prompt,
        used_prompt=system_message,
        few_shot_examples=few_shot_examples,
        test_results=eval_results,
        accuracy=accuracy,
    )


# =========================================================
# Ergebnisse speichern – Schema wie dein Zero-Shot-Beispiel
# =========================================================

def save_results_to_json(
    result: EvalRunResult,
    dataset_path: str,
    output_dir: str = "eval/few_shot_results",
) -> Path:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # ISO-Timestamp wie im Beispiel
    ts = datetime.now().isoformat()
    out_path = output_dir_path / f"fewshot_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # JSON-Schema an dein Beispiel angelehnt
    payload = {
        "timestamp": ts,
        "dataset": {
            "path": dataset_path,
            "num_examples": len(result.test_results),
        },
        "config": {
            "prompt": result.used_prompt,
            "model_name": result.model,
            "system_message_snippet": "",  # optional: result.used_prompt[:160]
            "mode": result.mode,  # hier "few-shot"
        },
        "result": {
            "prompt": result.used_prompt,
            "accuracy": result.accuracy,
            "results": result.test_results,
        },
    }

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_line(f"Saved JSON results to: {out_path.resolve()}")
    return out_path


# =========================================================
# Nutzung
# =========================================================

if __name__ == "__main__":
    # Pfade anpassen
    TRAIN_PATH = "navigate_200/dataset_random/navigate_40_train.json"
    TEST_PATH = "navigate_200/dataset_random/navigate_160_test.json"

    log_header("Loading datasets")
    train_data = load_dataset(TRAIN_PATH)
    test_data = load_dataset(TEST_PATH)
    log_line(f"Train examples: {len(train_data)}")
    log_line(f"Test  examples: {len(test_data)}")

    # Basis-System-Prompt (analog zu deinem Navigate-Task)
    base_system_prompt = (
        'You MUST answer with exactly one string: either "True" or "False". '
        'Do not output explanations, punctuation, or additional text. '
        '"True" = you return to the starting point. '
        '"False" = you do not return to the starting point.\n\n'
        "Additionally I am giving you following examples to learn from:\n\n"
    )

    # Few-Shot-Eval
    result = evaluate_few_shot(
        train_dataset=train_data,
        test_dataset=test_data,
        base_system_prompt=base_system_prompt,
        model="gpt-4o-mini",
        max_train_examples=40,
    )

    # JSON speichern – Ordner relativ: eval/few_shot_results
    save_results_to_json(
        result=result,
        dataset_path=TEST_PATH,
        output_dir="navigate_200/few_shot_results",
    )

    # Kurze Konsolen-Zusammenfassung
    print("\n\n========= FINAL RESULT (FEW-SHOT) =========")
    print("Accuracy:", f"{result.accuracy:.4f}")
    print("\nSample results (first 5):")
    for r in result.test_results[:5]:
        snippet = r["Input"].replace("\n", " ")[:80]
        print(
            f"Idx {r['Index']:3d}: acc={r['judgement']} | gold={r['correctAnswer']} | "
            f"model={r['MbaseAnswer']} | input={snippet}..."
        )
