"""
RAG-Evaluierung eines Prompts auf einem Klassifikations-Datensatz.

Ablauf analog zum Few-Shot-/Zero-Shot-Setup:
- JSON-Dataset einlesen
- Für jeden Index:
    - Frage + Ground-Truth-Label aus Dataset holen
    - Kontext aus Azure-Index abrufen (Azure Search)
    - System-Prompt = Base-Prompt + zusätzlicher Kontext
    - LLM aufrufen
    - Eval-Record bauen: {Index, Input, correctAnswer, MbaseAnswer, judgement, ...}
- Am Ende Accuracy berechnen
- Alles in JSON schreiben, Schema kompatibel zu Zero-/Few-Shot:
  {
    "timestamp": ...,
    "dataset": {..., "test_size": N, "split_id": ...},
    "config": {..., "mode": "rag"},
    "result": {"accuracy": ..., "results": [...]}
  }

Die Result-Files werden wie im Few-Shot-Skript benannt, z.B.:
  rag_eval_split5_train40_test160_YYYYMMDDHHMMSS.json
und liegen unter:
  {dataset_name}/rag_results/{split_id}/...
damit dein Analyse-Skript mit
  method_name_dir = "rag_results"
  method_name     = "rag"
arbeiten kann.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()


# =============================================================================
# Logging Helpers (wie in deinen anderen Skripten)
# =============================================================================

def log_header(title: str):
    print("\n" + "=" * 80)
    print(f"🔵 {title}")
    print("=" * 80)


def log_sub(title: str):
    print(f"\n--- {title} ---")


def log_line(msg: str):
    print(f"  • {msg}")


# =============================================================================
# Datentypen
# =============================================================================

@dataclass
class Example:
    index: int
    input: str
    label: str
    choices: Optional[List[str]] = None  # für Multiple-Choice-Datensätze


# =============================================================================
# OpenAI-Client + Retry-Wrapper
# =============================================================================

def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Setze die Umgebungsvariable OPENAI_API_KEY für OpenAI.")
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
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            print(
                f"⏳ Rate limit erreicht (Versuch {attempt+1}/{max_retries}), "
                f"warte {wait:.1f}s..."
            )
            time.sleep(wait)


# =============================================================================
# Normalisierung & Alignment (wie im Few-Shot-Skript)
# =============================================================================

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


# =============================================================================
# Azure Index Retrieval (AzureIndexRetriever-Äquivalent)
# =============================================================================

AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_API_KEY")


def retrieve_azure_context(azure_search_index: str, question: str, top_k: int = 3) -> str:
    """
    Fragt den Azure Search Index ab und baut einen RAG-Kontext bestehend
    aus Q/A-Beispielen (inputs + targets) — so wie Few-Shot-Beispiele.
    """
    AZURE_SEARCH_INDEX = azure_search_index

    if not (AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX and AZURE_SEARCH_API_KEY):
        log_line("⚠️ Azure Search nicht konfiguriert – Kontext bleibt leer.")
        return ""

    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2023-11-01"

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

    examples = []

    for d in docs:
        # Frage aus Index (field: inputs / content / text)
        q = d.get("inputs") or d.get("content") or d.get("text") or ""

        # Antwort aus Index (field: targets)
        targets = d.get("targets") or []
        if isinstance(targets, list) and targets:
            a = targets[0]
        else:
            a = ""

        if not q or not a:
            continue

        block = (
            "Example:\n"
            f"Question: {q}\n"
            f"Answer: {a}"
        )
        examples.append(block)

    context = "\n\n".join(examples)
    return context


# =============================================================================
# Dataset laden (generisch für verschiedene Klassifikations-Datasets)
# =============================================================================

def load_dataset(path: str) -> List[Example]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset: List[Example] = []

    for idx, item in enumerate(raw):
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
                if isinstance(item.get("targets"), list) and item.get("targets")
                else None
            )
            or None
        )

        # Optional: Multiple-Choice-Optionen
        choices = None
        mct = item.get("multiple_choice_targets") or item.get("choices")
        if isinstance(mct, list) and mct:
            choices = [str(c) for c in mct]

        if inp is None or label is None:
            print("⚠️ Warnung: Item konnte nicht geladen werden:", item)
            continue

        dataset.append(
            Example(
                index=idx,
                input=str(inp),
                label=str(label),
                choices=choices,
            )
        )

    return dataset


# =============================================================================
# RAG-Evaluierung
# =============================================================================

def evaluate_rag(
    dataset: List[Example],
    base_prompt: str,
    azure_search_index: str,
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Führt eine RAG-Evaluierung durch – pro Item:
      - Kontext aus Azure holen
      - System-Prompt = Base-Prompt + Kontext
      - LLM-Anfrage
      - Record wie im Few-Shot-Eval:
        {Index, Input, correctAnswer, MbaseAnswer, MbaseAnswerRaw, judgement, ...}
    """
    log_header("Starting RAG Evaluation")

    client = get_client()
    results: List[Dict[str, Any]] = []
    correct = 0

    log_sub("Dataset info")
    log_line(f"Number of examples: {len(dataset)}")

    for ex in dataset:
        # 1) Kontext vom Azure-Index holen
        context = retrieve_azure_context(azure_search_index, ex.input)

        # 2) System-Prompt bauen: Base-Prompt + Kontext + Hinweissatz
        if context:
            system_message = (
                base_prompt.rstrip()
                + "Additionally I am giving you following examples to learn from:\n\n"
                + context
                + "\n\nLearn from the given examples and answer the following question."
            )
        else:
            system_message = (
                base_prompt.rstrip()
            )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": ex.input},
        ]

        # 3) Anfrage ans LLM (mit Retry)
        resp = chat_with_retry(
            client,
            model=model_name,
            messages=messages,
            temperature=0.0,
        )
        time.sleep(0.1)  # leichte Drosselung

        answer_raw = (resp.choices[0].message.content or "").strip()
        gold_raw = ex.label.strip()

        # Antwort an Optionen ausrichten + normalisieren
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
            "Index": ex.index,
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

        log_line(
            f"Idx {ex.index}: raw={answer_raw} -> aligned={answer_aligned} "
            f"(norm={answer_norm}), gold={gold_raw} (norm={gold_norm}), ok={judgement}"
        )

    accuracy = correct / len(dataset) if dataset else 0.0
    log_sub("Evaluation finished")
    log_line(f"Accuracy = {accuracy:.4f}")

    return {
        "accuracy": accuracy,
        "results": results,
    }


# =============================================================================
# Ergebnisse speichern – Schema kompatibel zu Zero-/Few-Shot/Analyse-Skript
# =============================================================================

def save_rag_results(
    rag_eval: Dict[str, Any],
    *,
    dataset_path: str,
    output_dir: Path,
    train_size: int,
    test_size: int,
    split_id: str,
    model_name: str,
    base_system_prompt: str,
    azure_search_index: str,
) -> Path:
    """
    Speichert Ergebnis im gleichen Schema wie Few-Shot, nur mode="rag" und
    Dateiprefix "rag_eval_...".
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().isoformat()
    safe_ts = ts.replace(":", "").replace("-", "").replace(".", "")
    filename = (
        f"rag_eval_split{split_id}_train{train_size}_test{test_size}_{safe_ts}.json"
    )
    out_path = output_dir / filename

    payload = {
        "timestamp": ts,
        "dataset": {
            "path": dataset_path,
            "num_examples": len(rag_eval["results"]),
            "train_size": train_size,
            "test_size": test_size,
            "split_id": split_id,
        },
        "config": {
            "prompt": base_system_prompt,
            "model_name": model_name,
            "azure_search_index": azure_search_index,
            "system_message_snippet": "",  # optional
            "mode": "rag",
        },
        "result": {
            "prompt": base_system_prompt,
            "accuracy": rag_eval["accuracy"],
            "results": rag_eval["results"],
        },
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_line(f"Saved RAG JSON results to: {out_path.resolve()}")
    return out_path


# =============================================================================
# Nutzung – analog zum Few-Shot-Skript (Splits, Größen, Prompts)
# =============================================================================

if __name__ == "__main__":

    # Name des Datensatzes (wird im Pfad benutzt, analog Few-Shot)
    dataset_name = "elementary_math_qa"

    # Wurzelordner, in dem dataset_random/ liegt
    DATA_ROOT = dataset_name

    # Splits wie im Few-Shot-Skript
    test_numbers = ["5", "15", "19", "23", "31", "42", "48", "50", "56", "62"]
    test_numbers = ["31"]  # zum Debuggen

    # Testgrößen (N_test)
    TEST_SIZES = [40, 80, 120, 160]
    #TEST_SIZES = [120, 160]

    # Mapping: welches Train-Set (und Azure-Index) gehört zu welcher Testgröße?
    # -> existierende Indizes in Azure:
    #    "navigate_10_train", "navigate_20_train", "navigate_30_train", "navigate_40_train"
    TRAIN_SIZE_BY_TEST_SIZE = {
        40: 10,
        80: 20,
        120: 30,
        160: 40,
    }

    model_name = "gpt-4o-mini"

    # Prompt für Multiple-Choice (wie im Few-Shot-Skript)
    base_system_prompt_mcq = (
        "You are solving multiple-choice math questions.\n"
        "Each question lists several answer options, each prefixed with 'choice:'.\n"
        "Read the question and the choices carefully.\n"
        "Answer with exactly the text of the correct choice, without the 'choice:' "
        "prefix and without any explanation or extra text."
    )

    # Prompt für True/False-Klassifikation (Navigate)
    base_system_prompt_bool = (
        'You MUST answer with exactly one string: either "True" or "False". '
        'Do not output explanations, punctuation, or additional text. '
        '"True" = you return to the starting point. '
        '"False" = you do not return to the starting point.\n\n'
    )

    # >>> Wähle hier händisch den Prompt passend zum Datensatz <<<
    base_system_prompt = base_system_prompt_mcq
    # base_system_prompt = base_system_prompt_mcq

    for split_id in test_numbers:
        for test_size in TEST_SIZES:

            # train_size über Mapping bestimmen
            train_size = TRAIN_SIZE_BY_TEST_SIZE.get(test_size)
            if train_size is None:
                log_line(
                    f"⚠️ Keine Train-Size für test_size={test_size} definiert – skip."
                )
                continue

            print("\n" + "=" * 80)
            print(
                f"  RAG Evaluation – Split {split_id} – "
                f"Train {train_size} – Test {test_size}"
            )
            print("=" * 80)

            # Test-Pfad analog Few-Shot
            test_path = (
                f"{DATA_ROOT}/dataset_random/{split_id}/"
                f"{dataset_name}_{test_size}_test.json"
            )
            test_file = Path(test_path)

            if not test_file.exists():
                log_line(f"⚠️ Test-Datei nicht gefunden: {test_file} – skip.")
                continue

            # Azure-Indexname passend zur Train-Set-Größe
            AZURE_SEARCH_INDEX = f"{dataset_name}_{train_size}_train"
            # d.h. z.B. "navigate_10_train", "navigate_20_train", ...

            log_header("Loading dataset")
            test_data = load_dataset(str(test_file))
            log_line(f"Test examples: {len(test_data)}")

            if not test_data:
                log_line("⚠️ Leerer Test-Datensatz – skip.")
                continue

            rag_eval = evaluate_rag(
                dataset=test_data,
                base_prompt=base_system_prompt,
                azure_search_index=AZURE_SEARCH_INDEX,
                model_name=model_name,
            )

            # Ergebnisse speichern – Struktur analog zu Few-Shot
            results_dir = Path(f"{DATA_ROOT}/rag_results/{split_id}")
            save_rag_results(
                rag_eval=rag_eval,
                dataset_path=str(test_file),
                output_dir=results_dir,
                train_size=train_size,
                test_size=test_size,
                split_id=split_id,
                model_name=model_name,
                base_system_prompt=base_system_prompt,
                azure_search_index=AZURE_SEARCH_INDEX,
            )

            print(
                f"✔ Done: split={split_id}, train={train_size}, "
                f"test={test_size}, acc={rag_eval['accuracy']:.4f}"
            )
