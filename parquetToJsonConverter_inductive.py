import pandas as pd
import numpy as np
import json
from pathlib import Path

# Pfad zur Parquet-Datei
input_file = Path(
    r"C:\Users\kkrazius\Desktop\Bachelor-Arbeit\Datasets\eval\train-00000-of-00001 (6).parquet"
)

FILE_PREFIX = "navigate"

TRAIN_SIZES = [10, 20, 30, 40]
TEST_SIZES  = [50, 100, 150, 200]

TRAIN_TOTAL = 40         # Gesamtanzahl train samples
TEST_TOTAL  = 200        # Gesamtanzahl test samples

RANDOM_STATE = 42        # für Reproduzierbarkeit


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

    # Für deterministische Ergebnisse: erst sortieren, dann mischen
    df = df.sort_index().sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    total = len(df)
    required = TRAIN_TOTAL + TEST_TOTAL

    print(f"Gesamtanzahl Zeilen im Parquet: {total}")
    print(f"Benötigt für alle Subsets:       {required}")

    if total < required:
        raise ValueError(
            f"Nicht genug Daten: benötigt {required}, gefunden {total}"
        )

    print("Erstelle eindeutige Train- und Test-Basismengen …")

    # 40 für Train, danach 200 für Test
    train_base = df.iloc[:TRAIN_TOTAL].copy()
    test_base  = df.iloc[TRAIN_TOTAL:TRAIN_TOTAL + TEST_TOTAL].copy()

    out_dir = parquet_path.parent

    print("\n📘 TRAIN-Datensätze:")
    for n in TRAIN_SIZES:
        subset = train_base.iloc[:n]
        filename = f"{FILE_PREFIX}_{n}_train.json"
        export_json(subset, out_dir / filename)

    print("\n🧪 TEST-Datensätze:")
    for n in TEST_SIZES:
        subset = test_base.iloc[:n]
        filename = f"{FILE_PREFIX}_{n}_test.json"
        export_json(subset, out_dir / filename)

    print("\n✅ Fertig! Alle Datensätze wurden eindeutig und korrekt erzeugt.")


if __name__ == "__main__":
    main(input_file)
