import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_accuracy_from_judgement(items: List[Dict[str, Any]]) -> Tuple[float, int, int, int]:
    """
    Returns: (accuracy, num_correct, num_total, num_missing_judgement)
    - judgement==True counts as correct
    - judgement==False counts as incorrect
    - missing/None judgement is counted as missing (excluded from denominator)
    """
    correct = 0
    total = 0
    missing = 0

    for it in items:
        j = it.get("judgement", None)
        if j is True:
            correct += 1
            total += 1
        elif j is False:
            total += 1
        else:
            missing += 1

    acc = (correct / total) if total > 0 else 0.0
    return acc, correct, total, missing


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python accuracy_from_judgement.py <path_to_json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    data = load_json(path)

    results = (
        data.get("result", {}).get("results", [])
        if isinstance(data, dict)
        else []
    )

    if not isinstance(results, list) or not results:
        print("No results found at data['result']['results'].")
        sys.exit(2)

    acc, correct, total, missing = compute_accuracy_from_judgement(results)

    print(f"File: {path}")
    print(f"Judgement-based accuracy: {acc:.6f}  ({correct}/{total})")
    if missing:
        print(f"Missing/invalid judgement fields (excluded): {missing}")

    # Optional: compare against stored accuracy in JSON
    stored_acc = data.get("result", {}).get("accuracy", None)
    if isinstance(stored_acc, (int, float)):
        diff = acc - float(stored_acc)
        print(f"Stored accuracy in JSON: {float(stored_acc):.6f}")
        print(f"Difference (computed - stored): {diff:+.6f}")


if __name__ == "__main__":
    main()
