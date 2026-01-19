import json
import re
from copy import deepcopy

INPUT_FILE = r"C:\Users\Kevin\Desktop\Bachelor Thesis\bachelor-thesis-codebase\hf_eval_results_new\tasksource\bigbench_social_iqa_few-shot_new\validation\fewshot_social_iqa_validation_20251215_092116.json"
OUTPUT_FILE = r"C:\Users\Kevin\Desktop\Bachelor Thesis\bachelor-thesis-codebase\hf_eval_results_new\tasksource\bigbench_social_iqa_few-shot_new\validation\fewshot_social_iqa_validation_20251215_092116_fixed.json"


def normalize_answer(text: str) -> str:
    """
    Normalisiert Antworten, sodass semantisch gleiche Antworten
    übereinstimmen.
    """
    if not text:
        return ""

    text = text.lower()
    text = text.strip()

    # "answer:" entfernen
    text = re.sub(r"^answer\s*:\s*", "", text)

    # Mehrfache Leerzeichen reduzieren
    text = re.sub(r"\s+", " ", text)

    return text


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

fixed_count = 0

for item in data["result"]["results"]:
    if item.get("judgement") is False:
        correct = normalize_answer(item.get("correctAnswer", ""))
        model = normalize_answer(item.get("MbaseAnswer", ""))

        if correct == model:
            item["judgement"] = True
            fixed_count += 1

print(f"✔ Korrigierte Einträge: {fixed_count}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"📄 Gespeichert als: {OUTPUT_FILE}")
