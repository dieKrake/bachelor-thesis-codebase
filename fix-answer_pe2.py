import json
import re

INPUT_FILE = r"C:\Users\Kevin\Desktop\Bachelor Thesis\bachelor-thesis-codebase\pe2_hf_results\pe2_hf_elementary_math_qa_20260107_161118.json"
OUTPUT_FILE = r"C:\Users\Kevin\Desktop\Bachelor Thesis\bachelor-thesis-codebase\pe2_hf_results\pe2_hf_elementary_math_qa_20260107_161118_fixed.json"


def normalize_answer(text: str) -> str:
    """
    Normalisiert Antworten, sodass semantisch gleiche Antworten übereinstimmen.
    Entfernt typische Prefixes wie 'answer:' und 'choice:'.
    """
    if not text:
        return ""

    text = text.strip().lower()

    # Häufige Prefixes entfernen
    text = re.sub(r"^(answer|choice)\s*:\s*", "", text)

    # Mehrfache Leerzeichen reduzieren
    text = re.sub(r"\s+", " ", text)

    return text


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# In dieser Datei heißt die Liste "test_results" (nicht "results")
items = data.get("result", {}).get("test_results", [])

fixed_count = 0
fixed_indices = []

for item in items:
    if item.get("judgement") is False:
        correct = normalize_answer(item.get("correctAnswer", ""))

        # In der PE2-Datei: Modellantwort steht in MtaskAnswer (Raw optional)
        model = normalize_answer(item.get("MtaskAnswer", ""))

        if correct == model:
            item["judgement"] = True
            fixed_count += 1
            fixed_indices.append(item.get("Index"))

print(f"✔ Korrigierte Einträge: {fixed_count}")
if fixed_indices:
    print(f"📌 Indizes: {fixed_indices[:50]}" + (" ..." if len(fixed_indices) > 50 else ""))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"📄 Gespeichert als: {OUTPUT_FILE}")
