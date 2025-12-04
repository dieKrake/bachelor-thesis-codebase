"""
Zero-Shot Evaluation – angelehnt an den PE2-Algorithmus mit Logging.

- Lädt ein gelabeltes Dataset (Input + Label [+ optional choices]).
- Nutzt einen festen Prompt (Zero-Shot) zur Evaluation auf dem ganzen Datensatz.
- Verwendet dieselbe Mtask-Logik wie im PE2-Skript.
- Ist robust gegenüber:
    * True/False
    * Yes/No
    * Multiple-Choice (multiple_choice_targets + targets)
- Speichert Accuracy + Einzelergebnisse als JSON mit Timestamp & Dataset-Info.
"""

from __future__ import annotations

import os
import json
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
# Datentypen (erweitert um choices, kompatibel zu PE2)
# =========================================================

@dataclass
class Example:
    input: str
    label: str
    choices: Optional[List[str]] = None  # z.B. multiple_choice_targets


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
# Hilfsfunktion: Antworten/Labels normalisieren
# =========================================================

def normalize_answer(text: str) -> str:
    """
    Normalisiert Modellantworten und Labels, um verschiedene Schreibweisen
    robuster vergleichen zu können.

    - trimmt Whitespace
    - entfernt einfache Satzzeichen am Ende
    - macht lower()
    - schneidet führendes "choice:" ab (z.B. "choice:50" -> "50")
    - mappt True/False und Yes/No auf gemeinsame Repräsentation
    """
    if text is None:
        return ""

    t = str(text).strip()

    # ganz einfache Endzeichen weg (z.B. "True." -> "True")
    if t.endswith((".", "!", "?")):
        t = t[:-1]

    t = t.strip().lower()

    # Falls das Modell sowas wie "choice:50" oder "choice: 50" ausgibt:
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

    Strategie:
    - 'choice:'-Prefix entfernen (choice:50 -> 50)
    - Wenn die bereinigte Antwort EXAKT dem Goldlabel entspricht -> nimm das
    - Wenn Choices existieren:
        - prüfe EXAKTE Gleichheit mit einer Choice
        - OPTIONAL: versuche Tokens aus längeren Antworten mit Choices zu matchen
    - ansonsten: gib die bereinigte Antwort zurück
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

        # 3b) OPTIONAL: Token-basiert (für Antworten wie "I think the answer is 50.")
        # -> hier KEIN simples "if c in t", um Fälle wie "2" in "21" zu vermeiden
        tokens = t.replace(",", " ").replace(".", " ").split()
        for tok in tokens:
            for c in norm_choices:
                if tok == c:
                    return c

    # 4) Fallback: bereinigte Rohantwort zurückgeben
    return t


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

        answer_raw = (resp.choices[0].message.content or "").strip()
        gold_raw = ex.label.strip()

        # Antwort zuerst an die möglichen Optionen "ausrichten"
        answer_aligned = align_model_answer(
            answer_raw=answer_raw,
            gold_raw=gold_raw,
            choices=ex.choices,
        )

        # Normalisierung für robusteren Vergleich (True/False vs Yes/No etc.)
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
        }

        # Falls Multiple-Choice-Optionen vorhanden sind, mit ins Ergebnis schreiben
        if ex.choices is not None:
            record["choices"] = ex.choices

        eval_results.append(record)

        log_line(
            f"Example {idx}: model={answer_raw} (norm={answer_norm}), "
            f"gold={gold_raw} (norm={gold_norm}), ok={judgement}"
        )

    accuracy = correct / len(dataset) if dataset else 0.0
    log_line(f"Accuracy = {accuracy:.4f}")

    return PromptCandidate(
        prompt=candidate_prompt,
        accuracy=accuracy,
        eval_results=eval_results,
    )


# =========================================================
# Dataset laden (robuster für verschiedene Formate)
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
            or item.get("question")
            or None
        )

        # Label-Feld finden:
        # - label
        # - targets[0] (z.B. BigBench: korrektes multiple choice Target)
        # - correctAnswer
        # - answer
        # - output
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

        # Multiple-Choice-Optionen (falls vorhanden)
        choices = None
        mct = item.get("multiple_choice_targets") or item.get("choices")
        if isinstance(mct, list) and mct:
            choices = [str(c) for c in mct]

        if inp is None or label is None:
            print("⚠️ Warnung: Item konnte nicht geladen werden:", item)
            continue

        dataset.append(
            Example(
                input=str(inp),
                label=str(label),
                choices=choices,
            )
        )

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

    dataset_name = "elementary_math_qa"
    #dataset_name_split = dataset_name.split("_")[0]

    #test_numbers = ["5","15","19","23","31","42","48","50","56","62"]
    test_numbers = ["56"]
    #dataset_sizes = ["40", "80", "120", "160"]
    dataset_sizes = ["120","160"]

    # Prompt, Model, etc. bleiben wie bisher
    system_message_task = None
    model_name = "gpt-4o-mini"

    zero_shot_prompt_disabled = (
        'You MUST answer with exactly one string: either "True" or "False". '
        'Do not output explanations, punctuation, or additional text. '
        '"True" = you return to the starting point. '
        '"False" = you do not return to the starting point.\n\n'
        'Now answer the following question with "True" or "False":'
    )

    # Multiple-Choice-Prompt
    zero_shot_prompt = (
        "You are solving multiple-choice math questions.\n"
        "Each question lists several answer options, each prefixed with 'choice:'.\n"
        "Read the question and the choices carefully.\n"
        "Answer with exactly the text of the correct choice, without the 'choice:' prefix and without any explanation or extra text."
      )

    config = {
        "prompt": zero_shot_prompt,
        "model_name": model_name,
        "system_message_snippet": "",
        "mode": "zero-shot",
    }

    # ---------------------------------------------------------
    # LOOP ÜBER ALLE TESTORDNER UND ALLE DATASET-GRÖSSEN
    # ---------------------------------------------------------
    for test_number in test_numbers:
        for dataset_size in dataset_sizes:

            print("\n" + "="*40)
            print(f"  Test Set {test_number} – Size {dataset_size}")
            print("="*40)

            dataset_path = (
                f"{dataset_name}/dataset_random/{test_number}/"
                f"{dataset_name}_{dataset_size}_test.json"
            )

            log_header("Loading dataset")
            dataset = load_dataset(dataset_path)
            log_line(f"Loaded {len(dataset)} examples")

            # Zero-Shot Evaluation ausführen
            log_header("Running Zero-Shot Evaluation")
            result = zero_shot_evaluate(
                dataset=dataset,
                prompt=zero_shot_prompt,
                model_name=model_name,
                system_message=system_message_task,
            )

            # Ergebnis speichern
            results_dir = Path(f"{dataset_name}/zero_shot_results/{test_number}")
            results_dir.mkdir(parents=True, exist_ok=True)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = results_dir / f"zeroshot_eval_{dataset_size}_{timestamp_str}.json"

            save_run_to_json(
                output_path=output_file,
                dataset_path=dataset_path,
                dataset_size=len(dataset),
                config=config,
                result=result,
            )

            print(f"✔ Saved result for folder={test_number}, size={dataset_size}")
