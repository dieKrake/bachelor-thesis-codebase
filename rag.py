"""
RAG-Evaluierung eines Prompts auf einem Klassifikations-Datensatz (Navigate).

Ablauf analog zum Langflow-RAG-Flow:
- JSON-Dataset einlesen
- Für jeden Index:
    - Frage + Ground-Truth-Label aus Dataset holen
    - Kontext aus Azure-Index abrufen (AzureIndexRetriever)
    - System-Prompt = Base-Prompt + zusätzlicher Kontext
    - Mbase (OpenAI-Model) aufrufen
    - Eval-Record bauen: {Index, Input, correctAnswer, MbaseAnswer, judgement}
- Am Ende Accuracy berechnen
- Alles in JSON schreiben, Schema kompatibel zu Zero-/Few-Shot:
  {
    "timestamp": ...,
    "dataset": {...},
    "config": {..., "mode": "rag"},
    "result": {"accuracy": ..., "results": [...]}
  }

Dateiname wird so gewählt, dass dein Tabellen-Skript ihn findet:
navigate_<N>_results_rag.json
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import requests
from openai import OpenAI

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


# =============================================================================
# OpenAI-Client
# =============================================================================

def get_client() -> OpenAI:
    """
    Nutzt OPENAI_API_KEY aus der Umgebung.
    (Falls du wie in deinen anderen Skripten hart coden willst,
    kannst du das hier natürlich anpassen.)
    """
    api_key = os.envrion.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Setze die Umgebungsvariable OPENAI_API_KEY für OpenAI.")
    return OpenAI(api_key=api_key)


# =============================================================================
# Azure Index Retrieval (AzureIndexRetriever-Äquivalent)
# =============================================================================

AZURE_SEARCH_ENDPOINT = "https://ns-ai-index.search.windows.net"
AZURE_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_API_KEY")


def retrieve_azure_context(azure_search_index: str, question: str, top_k: int = 5) -> str:
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
        # Frage aus Index (field: inputs)
        q = d.get("inputs") or d.get("content") or d.get("text") or ""

        # Antwort aus Index (field: targets)
        targets = d.get("targets") or []
        if isinstance(targets, list) and targets:
            a = targets[0]
        else:
            a = ""

        if not q or not a:
            # Skip wenn eines von beiden fehlt
            continue

        block = (
            "Example:\n"
            f"Question: {q}\n"
            f"Answer: {a}"
        )
        examples.append(block)

    # Kontext zusammensetzen
    context = "\n\n".join(examples)

    return context



# =============================================================================
# Dataset laden
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
            or (item.get("targets")[0]
                if isinstance(item.get("targets"), list) and item["targets"] else None)
            or None
        )

        if inp is None or label is None:
            print("⚠️ Warnung: Item konnte nicht geladen werden:", item)
            continue

        dataset.append(Example(index=idx, input=str(inp), label=str(label)))

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
      - Record wie im BuildEvalJSONComponent:
        {Index, Input, correctAnswer, MbaseAnswer, judgement, context, system_message}
    """
    log_header("Starting RAG Evaluation")

    client = get_client()
    results: List[Dict[str, Any]] = []
    correct = 0

    log_sub("Dataset info")
    log_line(f"Number of examples: {len(dataset)}")

    for ex in dataset:
        # 1) Kontext vom Azure-Index holen (AzureIndexRetriever-Äquivalent)
        context = retrieve_azure_context(azure_search_index, ex.input)

        # 2) System-Prompt bauen (CombineTextInputs: base_prompt + additional_context)
        if context:
            system_message = (
                base_prompt.rstrip()
                + context
                + '\n\nLearn from the given examples and answer the following question with "True" or "False":'
            )
        else:
            # Fallback: wie Zero-Shot-Variante
            system_message = (
                base_prompt.rstrip()
                + '\n\nLearn from the given examples and answer the following question with "True" or "False":'
            )

        # 3) Anfrage ans LLM
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": ex.input},
        ]

        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
        )

        mbase_answer = (resp.choices[0].message.content or "").strip()
        gold = ex.label.strip()

        judgement = (mbase_answer == gold)
        if judgement:
            correct += 1

        # 4) Record inkl. Kontext speichern
        record = {
            "Index": ex.index,
            "Input": ex.input,
            "correctAnswer": gold,
            "MbaseAnswer": mbase_answer,
            "judgement": judgement,
            # NEU:
            "context": context,              # nur die aus Azure geholten Beispiele
            "system_message": system_message # kompletter Prompt, der ans Modell ging
        }
        results.append(record)

        log_line(
            f"Idx {ex.index}: mbase={mbase_answer}, gold={gold}, ok={judgement}"
        )

    accuracy = correct / len(dataset) if dataset else 0.0
    log_sub("Evaluation finished")
    log_line(f"Accuracy = {accuracy:.4f}")

    return {
        "accuracy": accuracy,
        "results": results,
        "prompt_used": base_prompt,
    }


# =============================================================================
# Ergebnisse speichern – Schema kompatibel zu Zero-/Few-Shot
# =============================================================================

def infer_n_from_dataset_path(dataset_path: str, default: Optional[int] = None) -> Optional[int]:
    """
    Erwartet z.B. 'navigate_200/dataset_random/navigate_100_test.json'
    und extrahiert die 100 als n.
    """
    m = re.search(r"navigate_(\d+)_test\.json", dataset_path)
    if m:
        return int(m.group(1))
    return default


def save_rag_results(
    dataset_path: str,
    model_name: str,
    base_prompt: str,
    rag_eval: Dict[str, Any],
    output_root: str = "navigate_200/rag_results",
) -> Path:
    """
    Speichert das Ergebnis als:
      navigate_<n>_results_rag.json
    in output_root.
    JSON-Schema:
    {
      "timestamp": ...,
      "dataset": {...},
      "config": {..., "mode": "rag"},
      "result": {...}
    }
    """
    examples = rag_eval["results"]
    accuracy = rag_eval["accuracy"]
    ts = datetime.now().isoformat()

    n = infer_n_from_dataset_path(dataset_path, default=len(examples))
    if n is None:
        n = len(examples)

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"navigate_{n}_results_rag.json"

    data = {
        "timestamp": ts,
        "dataset": {
            "path": dataset_path,
            "num_examples": len(examples),
        },
        "config": {
            "prompt": base_prompt,
            "model_name": model_name,
            "system_message_snippet": "",
            "mode": "rag",
        },
        "result": {
            "prompt": base_prompt,
            "accuracy": accuracy,
            "results": examples,
        },
    }

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log_line(f"Saved RAG JSON results to: {out_path}")
    return out_path


# =============================================================================
# Nutzung
# =============================================================================

if __name__ == "__main__":
    # Pfad zu deinem Test-Datensatz anpassen
    TEST_PATH = "navigate_200/dataset_random/navigate_160_test.json"
    AZURE_SEARCH_INDEX = "navigate_40_train"
    MODEL_NAME = "gpt-4o-mini"

    log_header("Loading dataset")
    test_data = load_dataset(TEST_PATH)
    log_line(f"Test examples: {len(test_data)}")

    # Base-System-Prompt analog zu Zero-/Few-Shot
    base_system_prompt = (
        'You MUST answer with exactly one string: either "True" or "False". '
        'Do not output explanations, punctuation, or additional text. '
        '"True" = you return to the starting point. '
        '"False" = you do not return to the starting point.\n\n'
        "Additionally I am giving you following examples to learn from:\n\n"
    )

    rag_eval = evaluate_rag(
        dataset=test_data,
        base_prompt=base_system_prompt,
        azure_search_index=AZURE_SEARCH_INDEX,
        model_name=MODEL_NAME,
    )

    save_rag_results(
        dataset_path=TEST_PATH,
        model_name=MODEL_NAME,
        base_prompt=base_system_prompt,
        rag_eval=rag_eval,
        output_root="navigate_200/rag_results",
    )

    print("\n\n========= FINAL RESULT (RAG) =========")
    print("Accuracy:", f"{rag_eval['accuracy']:.4f}")
    print("\nSample results (first 5):")
    for r in rag_eval["results"][:5]:
        snippet = r["Input"].replace("\n", " ")[:80]
        print(
            f"Idx {r['Index']:3d}: acc={r['judgement']} | gold={r['correctAnswer']} | "
            f"model={r['MbaseAnswer']} | input={snippet}..."
        )
