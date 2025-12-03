"""
PE2: Prompt Engineering a Prompt Engineer – Python-Implementierung mit Logging.

- Mtask: wertet einen Prompt auf einem gelabelten Datensatz aus (Accuracy).
- Mproposal: generiert neue Prompts basierend auf Fehlbeispielen.
- Beam-artige Suche: mehrere Prompt-Kandidaten parallel verfolgen.
"""

from __future__ import annotations

import random
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime  # ✅ neu: für Timestamp
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
    prompt: str
    accuracy: float = 0.0
    eval_results: List[Dict[str, Any]] = field(default_factory=list)


# =========================================================
# OpenAI-Client
# =========================================================

def get_client() -> OpenAI:
    # 🔒 Hier deinen API-Key eintragen oder über Umgebungsvariable lösen
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # <- hier openAI key eingeben


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
    log_sub("Evaluating prompt with Mtask")
    log_line(f"Prompt (first 100 chars): {candidate_prompt[:100]}...")

    client = get_client()
    eval_results: List[Dict[str, Any]] = []
    correct = 0

    for idx, ex in enumerate(dataset):
        user_content = f"{candidate_prompt}\n\n{ex.input}"

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_content})

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            #temperature=0.0,
        )

        answer = resp.choices[0].message.content.strip()
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

        log_line(f"Example {idx}: model={answer}, gold={gold}, ok={judgement}")

    accuracy = correct / len(dataset) if dataset else 0.0
    log_line(f"Accuracy = {accuracy:.4f}")

    return PromptCandidate(
        prompt=candidate_prompt,
        accuracy=accuracy,
        eval_results=eval_results,
    )


# =========================================================
# Hilfsfunktion: Fehlbeispiele formatieren
# =========================================================

def build_failure_examples_text(
    eval_results: List[Dict[str, Any]],
    failure_batch_size: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Sammelt Fehlbeispiele, sampelt ggf. und baut einen Textblock für den Meta-Prompt.
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
    eval_results: List[Dict[str, Any]],
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
        eval_results, failure_batch_size
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
        #temperature=0.0,
    )

    raw = resp.choices[0].message.content.strip()
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
# PE2-ähnlicher Beam-Algorithmus
# =========================================================

def pe2_optimize_prompt(
    dataset: List[Example],
    initial_prompt: str,
    meta_prompt_template: str,
    task_model: str = "gpt-4o-mini",
    proposal_model: str = "gpt-4o-mini",
    iterations: int = 3,
    beam_width: int = 3,          # Top-N Prompts pro Schritt
    proposals_per_prompt: int = 2,  # m: neue Prompts pro Kandidat
    failure_batch_size: int = 5,
    system_message_task: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Implementiert eine PE2-ähnliche Beam Search über Prompts.

    Rückgabe:
        {
          "best_prompt": str,
          "best_accuracy": float,
          "history": [
              {
                "iteration": int,
                "prompt": str,
                "accuracy": float
              },
              ...
          ]
        }
    """
    log_header("Starting PE2 Optimization")

    # Start mit einem Prompt
    current_candidates: List[PromptCandidate] = []

    # erste Evaluation des initialen Prompts
    base_candidate = run_mtask_on_dataset(
        candidate_prompt=initial_prompt,
        dataset=dataset,
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
            "accuracy": base_candidate.accuracy,
        }
    ]

    for it in range(1, iterations + 1):
        log_header(f"ITERATION {it}")

        # Sortiere derzeitige Kandidaten nach Accuracy
        current_candidates.sort(key=lambda c: c.accuracy, reverse=True)
        parents = current_candidates[:beam_width]

        log_line("Top prompts entering this iteration:")
        for p in parents:
            log_line(f"  acc={p.accuracy:.4f} | {p.prompt[:80]}...")

        # Early Stop, falls bereits perfekte Accuracy
        if parents and parents[0].accuracy >= 1.0:
            log_line("Perfect accuracy reached → stopping early")
            break

        new_candidates: List[PromptCandidate] = []

        # Für jeden Parent Prompt mehrere Vorschläge holen
        for parent in parents:
            for _ in range(proposals_per_prompt):
                log_sub("Generating improved prompt proposal")
                improved = run_mproposal(
                    current_prompt=parent.prompt,
                    eval_results=parent.eval_results,
                    meta_prompt_template=meta_prompt_template,
                    failure_batch_size=failure_batch_size,
                    model=proposal_model,
                )

                if not improved:
                    log_line("No valid improved prompt returned")
                    continue

                log_line(f"Received improved prompt: {improved[:80]}...")

                # Duplikate vermeiden
                if any(c.prompt == improved for c in current_candidates + new_candidates):
                    log_line("Improved prompt is duplicate → skipping")
                    continue

                # neuen Prompt auf Datensatz evaluieren
                cand = run_mtask_on_dataset(
                    candidate_prompt=improved,
                    dataset=dataset,
                    model=task_model,
                    system_message=system_message_task,
                )
                new_candidates.append(cand)

                history.append(
                    {
                        "iteration": it,
                        "prompt": cand.prompt,
                        "accuracy": cand.accuracy,
                    }
                )

                # global best updaten
                if cand.accuracy > best_accuracy:
                    best_accuracy = cand.accuracy
                    best_prompt = cand.prompt

        # Falls keine neuen Kandidaten generiert wurden → abbrechen
        if not new_candidates:
            log_line("No new candidates generated → stopping.")
            break

        # Nächste Runde:
        # kombiniere alte und neue, sortiere, nimm Top beam_width
        all_candidates = parents + new_candidates
        all_candidates.sort(key=lambda c: c.accuracy, reverse=True)
        current_candidates = all_candidates[:beam_width]

        log_sub("Beam selection results")
        for c in current_candidates:
            log_line(f"Keep acc={c.accuracy:.4f} | {c.prompt[:80]}...")

    log_header("PE2 Finished")
    log_line(f"Best Prompt (first 120 chars): {best_prompt[:120]}...")
    log_line(f"Best Accuracy: {best_accuracy:.4f}")

    return {
        "best_prompt": best_prompt,
        "best_accuracy": best_accuracy,
        "history": history,
    }


# =========================================================
# Dataset laden
# =========================================================

def load_dataset(path: str) -> list[Example]:
    raw = json.loads(Path(path).read_text())
    dataset = []

    for item in raw:
        # finde das Input-Feld
        inp = (
            item.get("input") or
            item.get("inputs") or
            item.get("Input") or
            item.get("instruction") or
            None
        )

        # finde das Label-Feld
        label = (
            item.get("label") or
            (item.get("targets")[0] if "targets" in item else None) or
            item.get("correctAnswer") or
            item.get("output") or
            None
        )

        if inp is None or label is None:
            print("⚠️ Warnung: Item konnte nicht geladen werden:", item)
            continue

        dataset.append(Example(input=inp, label=label))

    return dataset


# =========================================================
# Ergebnis in JSON speichern
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

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_header(f"Saved run to JSON: {output_path}")


# =========================================================
# Nutzung
# =========================================================

if __name__ == "__main__":
    # Pfad zum Dataset
    dataset_path = "navigate_200/dataset_random/navigate_40_test.json"

    log_header("Loading dataset")
    dataset = load_dataset(dataset_path)
    log_line(f"Loaded {len(dataset)} examples")

    # Initialer Prompt
    initial_prompt = (
        'You MUST answer with exactly one string: either "True" or "False". '
        'Do not output explanations, punctuation, or additional text. '
        '"True" = you return to the starting point. '
        '"False" = you do not return to the starting point.\n\n'
        'Now answer the following question with "True" or "False":'
    )

    # Meta-Prompt
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
    proposal_model = "gpt-5"

    config = {
        "initial_prompt": initial_prompt,
        "meta_prompt_template_snippet": meta_prompt_template[:200],
        "task_model": task_model,
        "proposal_model": proposal_model,
        "iterations": iterations,
        "beam_width": beam_width,
        "proposals_per_prompt": proposals_per_prompt,
        "failure_batch_size": failure_batch_size,
    }

    # PE2 ausführen
    result = pe2_optimize_prompt(
        dataset=dataset,
        initial_prompt=initial_prompt,
        meta_prompt_template=meta_prompt_template,
        task_model=task_model,
        proposal_model=proposal_model,
        iterations=iterations,
        beam_width=beam_width,
        proposals_per_prompt=proposals_per_prompt,
        failure_batch_size=failure_batch_size,
    )

    # Konsole: Zusammenfassung
    print("\n\n========= FINAL RESULT =========")
    print("Best Prompt:\n")
    print(result["best_prompt"])
    print("\nBest Accuracy:", result["best_accuracy"])

    print("\nHistory (iteration, accuracy, prompt snippet):")
    for h in result["history"]:
        snippet = h["prompt"].replace("\n", " ")[:100]
        print(f"Iter {h['iteration']}: acc={h['accuracy']:.3f} | {snippet}...")

    # JSON speichern
    results_dir = Path("pe2_results")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"pe2_run_{timestamp_str}.json"

    save_run_to_json(
        output_path=output_file,
        dataset_path=dataset_path,
        dataset_size=len(dataset),
        config=config,
        result=result,
    )
