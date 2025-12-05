"""
Few-Shot Evaluation – angelehnt an Zero-Shot- und PE2-Logik.

- Lädt Train- und Test-Datensätze (Input + Label [+ optional choices]).
- Train-Daten werden als Few-Shot-Beispiele (Q/A-Paare) in den System-Prompt eingebettet.
- Test-Daten werden mit diesem System-Prompt evaluiert.
- Robust gegenüber:
    * True/False
    * Yes/No
    * Multiple-Choice (multiple_choice_targets + targets)
- Speichert Accuracy + Einzelergebnisse als JSON mit Timestamp & Dataset-Info.
"""

from __future__ import annotations

import os
import json
import random
import time
from openai import OpenAI, RateLimitError
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
# Datentypen (kompatibel zum Zero-Shot-Skript)
# =========================================================

@dataclass
class Example:
    input: str
    label: str
    choices: Optional[List[str]] = None  # z.B. multiple_choice_targets


@dataclass
class EvalRunResult:
    mode: str  # "few-shot"
    model: str
    base_system_prompt: str
    used_prompt: str  # tatsächlicher System-Prompt inkl. Few-Shot-Beispiele
    few_shot_examples: List[Example] = field(default_factory=list)
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    accuracy: float = 0.0


# =========================================================
# OpenAI-Client
# =========================================================

def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Bitte OPENAI_API_KEY in der Umgebung setzen.")
    return OpenAI(api_key=api_key)


# =========================================================
# Normalisierung & Alignment wie im Zero-Shot-Skript
# =========================================================

def normalize_answer(text: str) -> str:
    """
    Normalisiert Modellantworten und Labels, um verschiedene Schreibweisen
    robuster vergleichen zu können.
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
# Dataset laden (robust für verschiedene Formate)
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
            or item.get("instruction")
            or item.get("question")
            or None
        )

        # Label-Feld suchen:
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
# Few-Shot-Kontext bauen
# =========================================================

def build_few_shot_block(
    train_dataset: List[Example],
    max_examples: int = 8,
) -> List[Example]:
    """Wählt bis zu max_examples Beispiele aus dem Train-Set (zufällig)."""
    if not train_dataset:
        return []
    if len(train_dataset) <= max_examples:
        return list(train_dataset)
    return random.sample(train_dataset, max_examples)


def format_few_shot_block(examples: List[Example]) -> str:
    """
    Formatiert Few-Shot-Beispiele als Textblock.
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
        except RateLimitError as e:
            # beim letzten Versuch Fehler weiterwerfen
            if attempt == max_retries - 1:
                raise

            wait = base_delay * (2 ** attempt)
            print(
                f"⏳ Rate limit erreicht (Versuch {attempt+1}/{max_retries}), "
                f"warte {wait:.1f}s..."
            )
            time.sleep(wait)


# =========================================================
# Few-Shot-Evaluierung (nutzt normalize/align wie Zero-Shot)
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

    # Few-Shot-Beispiele wählen
    log_sub("Selecting few-shot examples")
    few_shot_examples = build_few_shot_block(
        train_dataset, max_examples=max_train_examples
    )
    log_line(f"Using {len(few_shot_examples)} train examples as few-shot context")

    few_shot_text = format_few_shot_block(few_shot_examples)

    # System-Prompt aufbauen
    system_message = (
        base_system_prompt.strip()
        + "Additionally I am giving you following examples to learn from:\n\n"
        + few_shot_text
        + "\n\nLearn from the given examples and answer the following question."
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
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": ex.input},
        ]

        resp = chat_with_retry(
            client,
            model=model,
            messages=messages,
            temperature=0.0,
        )
        # optional: leicht drosseln, um gar nicht erst ans Limit zu kommen
        time.sleep(0.1)  # 100ms

        answer_raw = (resp.choices[0].message.content or "").strip()
        gold_raw = ex.label.strip()

        # Antwort zuerst an die möglichen Optionen "ausrichten"
        answer_aligned = align_model_answer(
            answer_raw=answer_raw,
            gold_raw=gold_raw,
            choices=ex.choices,
        )

        # Normalisierung (True/False vs Yes/No etc.)
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
        if ex.choices is not None:
            record["choices"] = ex.choices

        eval_results.append(record)

        log_line(
            f"Test {idx}: model_raw={answer_raw} -> aligned={answer_aligned} "
            f"(norm={answer_norm}), gold={gold_raw} (norm={gold_norm}), ok={judgement}"
        )

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
# Ergebnisse speichern – Schema kompatibel zur Zero-Shot-Analyse
# =========================================================

def save_results_to_json(
    result: EvalRunResult,
    *,
    dataset_path: str,
    output_dir: Path,
    train_size: int,
    test_size: int,
    split_id: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().isoformat()
    safe_ts = ts.replace(":", "").replace("-", "").replace(".", "")
    filename = f"fewshot_eval_split{split_id}_train{train_size}_test{test_size}_{safe_ts}.json"
    out_path = output_dir / filename

    payload = {
        "timestamp": ts,
        "dataset": {
            "path": dataset_path,
            "num_examples": len(result.test_results),
            "train_size": train_size,
            "test_size": test_size,
            "split_id": split_id,
        },
        "config": {
            "prompt": result.base_system_prompt,
            "model_name": result.model,
            "system_message_snippet": result.used_prompt[:200],
            "mode": result.mode,  # "few-shot"
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
# Nutzung / Main – 20/80-Skalierung von Train- und Test-Size
# =========================================================

if __name__ == "__main__":

    dataset_name = "navigate"

    # Splits wie beim Zero-Shot
    test_numbers = ["5","15", "19", "23", "31", "42", "48", "50", "56", "62"]
    #test_numbers = ["5"]  # zum Debuggen

    # Testgrößen (N_test)
    TEST_SIZES = [40, 80, 120, 160]
    #TEST_SIZES = [160]

    # gewünschtes Verhältnis: train : test = 20 : 80 = 1 : 4
    TRAIN_TEST_RATIO = 0.25  # train_size = test_size * 0.25

    model_name = "gpt-4o-mini"

    base_system_prompt_disabled = (
        "You are solving multiple-choice math questions.\n"
        "Each question lists several answer options, each prefixed with 'choice:'.\n"
        "Read the question and the choices carefully.\n"
        "Answer with exactly the text of the correct choice, without the 'choice:' "
        "prefix and without any explanation or extra text."
    )

    base_system_prompt = (
        'You MUST answer with exactly one string: either "True" or "False". '
        'Do not output explanations, punctuation, or additional text. '
        '"True" = you return to the starting point. '
        '"False" = you do not return to the starting point.\n\n'
    )

    for split_id in test_numbers:
        for test_size in TEST_SIZES:

            # train_size entsprechend dem 20/80-Verhältnis
            train_size = int(test_size * TRAIN_TEST_RATIO)

            print("\n" + "=" * 80)
            print(
                f"  Few-Shot Evaluation – Split {split_id} – "
                f"Train {train_size} – Test {test_size}"
            )
            print("=" * 80)

            # Pfade (Train & Test im gleichen Ordner wie bei Zero-Shot)
            train_path = (
                f"{dataset_name}/dataset_random/{split_id}/"
                f"{dataset_name}_{train_size}_train.json"
            )
            test_path = (
                f"{dataset_name}/dataset_random/{split_id}/"
                f"{dataset_name}_{test_size}_test.json"
            )

            train_file = Path(train_path)
            test_file = Path(test_path)

            # Existenz prüfen, sonst überspringen
            if not train_file.exists():
                log_line(f"⚠️ Train-Datei nicht gefunden: {train_file} – skip.")
                continue
            if not test_file.exists():
                log_line(f"⚠️ Test-Datei nicht gefunden: {test_file} – skip.")
                continue

            log_header("Loading datasets")
            train_data = load_dataset(str(train_file))
            test_data = load_dataset(str(test_file))
            log_line(f"Train examples: {len(train_data)}")
            log_line(f"Test  examples: {len(test_data)}")

            if not train_data or not test_data:
                log_line("⚠️ Leerer Train- oder Test-Datensatz – skip.")
                continue

            # Few-Shot-Eval
            result = evaluate_few_shot(
                train_dataset=train_data,
                test_dataset=test_data,
                base_system_prompt=base_system_prompt,
                model=model_name,
                max_train_examples=train_size,  # passt zur Train-Filegröße
            )

            # Ergebnisse speichern – Struktur analog zu Zero-Shot
            results_dir = Path(f"{dataset_name}/few_shot_results/{split_id}")
            save_results_to_json(
                result,
                dataset_path=str(test_file),
                output_dir=results_dir,
                train_size=train_size,
                test_size=test_size,
                split_id=split_id,
            )

            print(
                f"✔ Done: split={split_id}, train={train_size}, "
                f"test={test_size}, acc={result.accuracy:.4f}"
            )
