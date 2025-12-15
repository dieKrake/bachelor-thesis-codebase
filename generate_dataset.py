"""
Helper-Skript: Erzeuge ein lokales JSON-Dataset aus HuggingFace

Unterstützt u.a.:
- BIG-Bench:   HF_DATASET_NAME = "tasksource/bigbench", HF_TASK_NAME = "<task>"
- BoolQ:       HF_DATASET_NAME = "google/boolq", HF_TASK_NAME = None

Allgemeine Idee:
- Lädt den gewünschten HF-Split (z.B. "train", "validation").
- Nimmt die ersten N Beispiele.
- Schreibt sie als JSON-Liste auf die Platte.

Das JSON ist kompatibel mit deinen Few-/Zero-Shot-/RAG-Skripten, denn:
- jedes Item enthält mindestens:
    - "input": String (Frage / Prompt)
    - "label": String (korrekte Antwort, z.B. "True"/"False")
- falls vorhanden:
    - "multiple_choice_targets": Liste von Strings
    - "choices":                 gleiche Liste (für Loader-Kompatibilität)
- optional: Originalfelder aus dem HF-Dataset
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset as hf_load_dataset


# =========================================================
# 🔧 KONFIGURATION
# =========================================================

# Beispiel 1: BIG-Bench Logical Fallacy
# HF_DATASET_NAME = "tasksource/bigbench"
# HF_TASK_NAME = "logical_fallacy_detection"

# Beispiel 2: BoolQ (kein Task-Name)
# HF_DATASET_NAME = "google/boolq"
# HF_TASK_NAME = None

HF_DATASET_NAME = "tasksource/bigbench"      # oder "google/boolq"
HF_TASK_NAME = "fact_checker"   # oder None für boolq

HF_SPLIT = "train"                           # z.B. "train", "validation", "test"
NUM_EXAMPLES = 100                           # gewünschte Anzahl Beispiele

OUTPUT_DIR = Path("generated_datasets")

# Basisname für die Output-Datei bestimmen
if HF_TASK_NAME in (None, "", "None"):
    base_name = HF_DATASET_NAME.split("/")[-1]   # z.B. "boolq"
else:
    base_name = HF_TASK_NAME                     # z.B. "logical_fallacy_detection"

OUTPUT_FILE = OUTPUT_DIR / f"{base_name}_{HF_SPLIT}_{NUM_EXAMPLES}_train.json"


# =========================================================
# HF → JSON-Konvertierung
# =========================================================

def main():
    # -----------------------------------------------------
    # Dataset laden
    # -----------------------------------------------------
    if HF_TASK_NAME in (None, "", "None"):
        # z.B. google/boolq
        print(f"🔵 Lade HF-Dataset: {HF_DATASET_NAME}[{HF_SPLIT}]")
        ds = hf_load_dataset(HF_DATASET_NAME, split=HF_SPLIT)
    else:
        # z.B. tasksource/bigbench/logical_fallacy_detection
        print(f"🔵 Lade HF-Dataset: {HF_DATASET_NAME}/{HF_TASK_NAME}[{HF_SPLIT}]")
        ds = hf_load_dataset(HF_DATASET_NAME, HF_TASK_NAME, split=HF_SPLIT)

    available = len(ds)
    print(f"  • Verfügbare Beispiele im Split: {available}")

    n = min(NUM_EXAMPLES, available)
    if n < NUM_EXAMPLES:
        print(f"  ⚠️ Gewünschte {NUM_EXAMPLES} > {available}, verwende {n} Beispiele.")

    ds = ds.select(range(n))

    converted = []

    # -----------------------------------------------------
    # Konvertierung pro Beispiel
    # -----------------------------------------------------
    for idx, item in enumerate(ds):
        # Spezielle Behandlung: BoolQ
        if HF_DATASET_NAME == "google/boolq":
            # BoolQ-Felder: question (str), passage (str), answer (bool)
            question = item["question"]
            passage = item["passage"]
            answer = item["answer"]

            # Genau dieselbe Input/Label-Definition wie im Eval-Skript
            inp = f"Question: {question}\nPassage: {passage}"
            label = "True" if bool(answer) else "False"
            choices = None

            out_item = {
                "input": str(inp),
                "label": str(label),
                # optionale Originalfelder für Debugging
                "question": question,
                "passage": passage,
                "answer": bool(answer),
            }

        # Standard-Pfad: BIG-Bench & Co.
        else:
            # ---------------- Input ----------------
            inp = (
                item.get("input")
                or item.get("inputs")
                or item.get("question")
                or item.get("instruction")
                or None
            )

            # ---------------- Label ----------------
            targets = item.get("targets") or []
            label = (
                item.get("label")
                or (targets[0] if isinstance(targets, list) and targets else None)
                or item.get("answer")
                or item.get("output")
                or None
            )

            # ---------------- Choices / MC-Targets ----------------
            mct = item.get("multiple_choice_targets") or item.get("choices")
            choices = None
            if isinstance(mct, list) and mct:
                choices = [str(c) for c in mct]

            if inp is None or label is None:
                print(f"  ⚠️ Beispiel {idx} ohne input/label – übersprungen.")
                continue

            out_item = {
                # Felder, die deine Few-/Zero-/RAG-Skripte sicher verstehen
                "input": str(inp),
                "label": str(label),
            }

            if choices is not None:
                # doppelt, weil deine Loader sowohl multiple_choice_targets
                # als auch choices berücksichtigen
                out_item["multiple_choice_targets"] = choices
                out_item["choices"] = choices

            # Optionale Originalfelder (schaden nicht, helfen beim Debuggen)
            if "inputs" in item:
                out_item["inputs"] = item["inputs"]
            if "targets" in item and item["targets"] is not None:
                out_item["targets"] = item["targets"]
            if "idx" in item:
                out_item["idx"] = item["idx"]

        converted.append(out_item)

    # -----------------------------------------------------
    # Speichern
    # -----------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n✅ Fertig!")
    print(f"  • Gespeicherte Beispiele: {len(converted)}")
    print(f"  • Datei: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
