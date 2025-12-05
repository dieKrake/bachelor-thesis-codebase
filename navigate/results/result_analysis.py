import pandas as pd
import json
from pathlib import Path
import re

# --- Hilfsfunktion: Accuracy robust extrahieren ---
def extract_accuracy(data):
    """
    Versucht, accuracy aus verschiedenen JSON-Strukturen zu extrahieren.
    Unterstützt z.B.:
    - {"accuracy": 0.58, ...}
    - {"result": {"accuracy": 0.695, ...}, ...}
    - beliebig verschachtelte Dicts/Listen mit einem "accuracy"-Key
    """
    # 1) Direkter Zugriff auf Top-Level
    if isinstance(data, dict):
        if "accuracy" in data and isinstance(data["accuracy"], (int, float)):
            return data["accuracy"]

        # 2) Spezieller Fall: result.accuracy
        if "result" in data and isinstance(data["result"], dict):
            res_acc = extract_accuracy(data["result"])
            if res_acc is not None:
                return res_acc

        # 3) Allgemeine rekursive Suche in allen Dict-Werten
        for value in data.values():
            acc = extract_accuracy(value)
            if acc is not None:
                return acc

    # 4) Falls data eine Liste ist: in allen Elementen suchen
    if isinstance(data, list):
        for item in data:
            acc = extract_accuracy(item)
            if acc is not None:
                return acc

    # Nichts gefunden
    return None


# --- Tabellenstruktur vorbereiten ---
rows = [
    r"bigbench/navigate\\n = 40",
    r"bigbench/navigate\\n = 80",
    r"bigbench/navigate\\n = 120",
    r"bigbench/navigate\\n = 160",
]

# Mapping von Methoden zu Spaltennamen
columns = {
    "zero": "Zero-Shot",
    "zero-shot": "Zero-Shot",
    "few": "Few-Shot",
    "few-shot": "Few-Shot",
    "rag": "RAG",
    "pe2": "PE2",
}

column_order = ["Zero-Shot", "Few-Shot", "RAG", "PE2"]

df = pd.DataFrame(index=rows, columns=column_order)

# --- Pfad zu deinen JSON-Files ---
# Passe das ggf. an: z.B. Path("eval/zero_shot_results") etc.
results_path = Path("navigate_200/results")

# Optional: alter Pattern-Fallback für alte Dateinamen
old_name_pattern = re.compile(r"navigate_(\d+)_results_(zero|few|rag|pe2)\.json")

# --- Alle JSON Dateien durchlaufen ---
for file in results_path.glob("*.json"):
    # LEERDATEI-PRÜFUNG
    if file.stat().st_size == 0:
        print(f"⚠️ Datei ist leer: {file.name} (0 Bytes) – übersprungen")
        continue

    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Fehler: Datei {file.name} enthält kein gültiges JSON – übersprungen")
        continue

    # --- Accuracy robust extrahieren ---
    accuracy = extract_accuracy(data)
    if accuracy is None:
        print(f"⚠️ Keine Accuracy in Datei gefunden: {file.name} – übersprungen")
        continue

    # --- n (Anzahl Beispiele) bestimmen ---
    n_value = None

    # Neuer Stil: dataset.num_examples
    if isinstance(data, dict) and "dataset" in data and isinstance(data["dataset"], dict):
        n_value = data["dataset"].get("num_examples")

    # Fallback: versuche aus dataset.path zu parsen: ...navigate_100_test.json
    if n_value is None and "dataset" in data and isinstance(data["dataset"], dict):
        ds_path = data["dataset"].get("path", "")
        m = re.search(r"navigate_(\d+)_test", ds_path)
        if m:
            n_value = int(m.group(1))

    # Noch ein Fallback: alter Dateiname (navigate_100_results_zero.json)
    if n_value is None:
        m = old_name_pattern.match(file.name)
        if m:
            n_value = int(m.group(1))

    if n_value is None:
        print(f"⚠️ Konnte n (Anzahl Beispiele) nicht bestimmen für Datei: {file.name}")
        continue

    # --- Methode bestimmen ---
    method_key = None

    # Neuer Stil: config.mode (z.B. "zero-shot", "few-shot")
    if "config" in data and isinstance(data["config"], dict):
        mode = data["config"].get("mode")
        if isinstance(mode, str):
            method_key = mode.strip()

    # Fallback: alter Dateiname
    if method_key is None:
        m = old_name_pattern.match(file.name)
        if m:
            method_key = m.group(2)

    if method_key is None:
        print(f"⚠️ Konnte Methode (mode) nicht bestimmen für Datei: {file.name}")
        continue

    # Auf bekannte Methoden mappen
    if method_key not in columns:
        print(f"⚠️ Unbekannte Methode '{method_key}' in Datei: {file.name}")
        continue

    col_label = columns[method_key]

    # Zeilenlabel wie vorher: bigbench/navigate\\n = X
    row_label = f"bigbench/navigate\\\\n = {n_value}"

    if row_label not in df.index:
        print(f"⚠️ n={n_value} nicht in vordefinierten Zeilen – dennoch eingetragen.")
        # Optional: Zeile dynamisch hinzufügen
        df.loc[row_label] = {c: None for c in df.columns}

    # Accuracy auf 2 Nachkommastellen runden
    accuracy_rounded = round(accuracy, 2)

    # Wert eintragen (überschreibt ggf. alte Werte)
    df.loc[row_label, col_label] = accuracy_rounded
    print(f"✅ Eingetragen: Datei={file.name}, n={n_value}, mode={method_key}, acc={accuracy_rounded}")

# --- Tabelle speichern ---
output_file = results_path / "accuracy_table.csv"
df.to_csv(output_file)

print("\nTabelle generiert:")
print(df)
print(f"\nCSV gespeichert unter: {output_file.resolve()}")
