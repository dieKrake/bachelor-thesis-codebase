import pandas as pd
import numpy as np
import json
from pathlib import Path

# Pfad zur Parquet-Datei
DATASET_NAME = "logical_fallacy_detection"

DATASET_PATH = Path(r"C:\Users\kkrazius\Desktop\Bachelor-Arbeit\Datasets\eval")
INPUT_FILE = DATASET_PATH / f"{DATASET_NAME}.parquet"

FILE_PREFIX = DATASET_NAME

TEST_SIZES  = [200]
TRAIN_SIZES = [10, 20, 30, 40]

RANDOM_STATE =  49 # für Reproduzierbarkeit


def make_serializable(obj):
    """Wandelt Numpy/Pandas-Typen in JSON-freundliche Python-Typen um."""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()

    if isinstance(obj, np.generic):  # z.B. numpy.int32, float64, bool_
        return obj.item()

    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]

    if isinstance(obj, (pd.Timestamp, )):
        return obj.isoformat()

    return obj


def export_json(df, path: Path):
    df = df.reset_index(drop=True)
    records = df.to_dict(orient="records")
    serializable_records = [make_serializable(r) for r in records]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_records, f, ensure_ascii=False, indent=2)

    print(f"  → {path} ({len(serializable_records)} Elemente)")


def main(parquet_path: Path):
    print(f"Lade {parquet_path} …")
    df = pd.read_parquet(parquet_path)

    total = len(df)
    required = sum(TEST_SIZES) + sum(TRAIN_SIZES)  # = 600
    print(f"Gesamtanzahl Zeilen im Parquet: {total}")
    print(f"Benötigt für alle disjunkten Subsets: {required}")

    if total < required:
        raise ValueError(
            f"Nicht genug Daten: benötigt {required}, gefunden {total}"
        )

    # Zufällig 600 eindeutige Zeilen ziehen
    # (ohne Zurücklegen, also garantiert unique)
    sampled = df.sample(n=required, random_state=RANDOM_STATE).reset_index(drop=True)

    out_dir = parquet_path.parent

    offset = 0

    print("\n🧪 TEST-Datensätze (disjunkt):")
    for n in TEST_SIZES:
        subset = sampled.iloc[offset:offset + n].copy()
        offset += n

        filename = f"{FILE_PREFIX}_{n}_test.json"
        export_json(subset, out_dir / filename)

    print("\n📘 TRAIN-Datensätze (disjunkt):")
    for n in TRAIN_SIZES:
        subset = sampled.iloc[offset:offset + n].copy()
        offset += n

        filename = f"{FILE_PREFIX}_{n}_train.json"
        export_json(subset, out_dir / filename)

    print("\n✅ Fertig! 8 disjunkte Datensätze aus 600 zufälligen Elementen erzeugt.")


if __name__ == "__main__":
    main(INPUT_FILE)
