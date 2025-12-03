import pandas as pd
import numpy as np
import json
from pathlib import Path

# === PARAMETER ===========================================================
# Pfad zur Parquet-Datei
INPUT_FILE = Path(
    r"C:\Users\kkrazius\Desktop\Bachelor-Arbeit\Datasets\eval\train-00000-of-00001 (6).parquet"
)

OUTPUT_FILENAME = "navigate_full.json"   # Name der Ausgabe-JSON
# ========================================================================


def make_serializable(obj):
    """Wandelt Numpy/Pandas-Typen in JSON-freundliche Python-Typen um."""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    return obj


def export_json(df, path: Path):
    """Speichert einen kompletten DataFrame als JSON-Liste."""
    df = df.reset_index(drop=True)
    records = df.to_dict(orient="records")

    # rekursiv serialisieren
    serializable_records = [make_serializable(r) for r in records]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_records, f, ensure_ascii=False, indent=2)

    print(f"📄 JSON gespeichert: {path} ({len(serializable_records)} Elemente)")


def main(parquet_path: Path):
    print(f"Lade Parquet-Datei: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"➡ Anzahl Zeilen: {len(df)}")

    # Ausgabeordner = Ordner der Parquet-Datei
    output_path = parquet_path.parent / OUTPUT_FILENAME

    export_json(df, output_path)

    print("\n✅ Fertig! Gesamter Datensatz wurde als JSON exportiert.")


if __name__ == "__main__":
    main(INPUT_FILE)
