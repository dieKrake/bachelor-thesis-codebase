import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# Wurzel ist der Ordner, in dem das Skript liegt (z. B. hf_eval_results_new)
ROOT_DIR = Path(__file__).resolve().parent


def load_metrics_from_file(file_path: Path):
    """
    Lädt Accuracy + Testgröße (N) aus einer JSON-Datei.
    Erwartetes Schema:
      "dataset": { "test_size": ..., ... }
      "result":  { "accuracy": ... }

    Gibt (N, accuracy) zurück oder (None, None) bei Problemen.
    """
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ Konnte {file_path} nicht lesen/parsen: {e}")
        return None, None

    try:
        accuracy = data["result"]["accuracy"]
    except Exception as e:
        print(f"⚠️ Keine 'result.accuracy' in {file_path}: {e}")
        return None, None

    dataset_info = data.get("dataset", {})
    test_size = dataset_info.get("test_size", None)

    # Falls keine test_size im JSON: wie bei dir standardmäßig 200
    if test_size is None:
        test_size = 200
    else:
        test_size = int(test_size)

    return test_size, float(accuracy)


def parse_dataset_and_method(component_name: str):
    """
    component_name ist z. B. 'navigate_rag_new' oder
    'elementary_math_qa_zero-shot_new' oder 'boolq_few-shot_new'.

    Heuristik:
      - split an '_'
      - Methode = letzte zwei Teile (z. B. 'rag_new', 'few-shot_new')
      - Dataset = Rest vorne (z. B. 'navigate', 'elementary_math_qa', 'boolq')
    """
    parts = component_name.split("_")
    if len(parts) >= 3:
        method = "_".join(parts[-2:])
        dataset = "_".join(parts[:-2])
    elif len(parts) == 2:
        dataset, method = parts[0], parts[1]
    else:
        dataset = component_name
        method = "unknown"

    return dataset, method


def collect_all_results(root: Path) -> pd.DataFrame:
    """
    Läuft rekursiv durch den Wurzelordner und sammelt alle JSON-Dateien ein.
    Gibt einen DataFrame mit Spalten:
      ['dataset', 'method', 'N', 'accuracy', 'file']
    zurück.
    """

    rows = []

    print(f"Starte rekursive Suche nach JSON-Dateien unter: {root}")

    for file in root.rglob("*.json"):
        if not file.is_file():
            continue

        N, acc = load_metrics_from_file(file)
        if N is None or acc is None:
            # Datei ist nicht im erwarteten Schema
            continue

        # Relevante Komponente für Dataset+Methode bestimmen:
        #   .../<dataset_method>/validation/file.json
        # oder bei google:
        #   .../google/<dataset_method>/validation/file.json
        parent = file.parent
        if parent.name == "validation" and parent.parent != root:
            comp = parent.parent
        else:
            comp = parent

        component_name = comp.name
        dataset, method = parse_dataset_and_method(component_name)

        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "N": int(N),
                "accuracy": acc,
                "file": str(file.relative_to(root)),
            }
        )

    if not rows:
        print("❌ Keine verwertbaren JSON-Dateien gefunden.")
        return pd.DataFrame(columns=["dataset", "method", "N", "accuracy", "file"])

    df = pd.DataFrame(rows)
    output_csv = root / "overall_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"→ Gesamtergebnis als CSV gespeichert: {output_csv}")

    return df


def plot_bar_charts(all_results: pd.DataFrame, root: Path):
    """
    Erstellt für jedes Dataset ein Balkendiagramm:
    x-Achse: Methoden (few-shot_new, zero-shot_new, rag_new, ...)
    y-Achse: mittlere Accuracy (über alle Runs dieser Methode/Dataset).
    Zusätzlich wird eine Summary-CSV erzeugt.
    """

    if all_results.empty:
        print("⚠️ all_results ist leer – keine Plots möglich.")
        return

    # Summary pro Dataset+Methode
    summary = (
        all_results
        .groupby(["dataset", "method"])["accuracy"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
    )

    summary_csv = root / "summary_by_dataset_method.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"→ Summary (Dataset/Method) gespeichert: {summary_csv}")

    # Für jedes Dataset ein Balkendiagramm
    for dataset, df_ds in summary.groupby("dataset"):
        methods = df_ds["method"].tolist()
        means = df_ds["mean"].tolist()

        x = range(len(methods))

        plt.figure(figsize=(max(6, len(methods) * 0.8), 5))
        plt.bar(x, means)

        plt.xticks(x, methods, rotation=45, ha="right")
        plt.xlabel("Methode")
        plt.ylabel("Accuracy (Mittelwert)")
        plt.title(f"Accuracy pro Methode – Dataset: {dataset}")
        plt.ylim(0.0, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        output = root / f"{dataset}_barplot.png"
        plt.tight_layout()
        plt.savefig(output, bbox_inches="tight")
        plt.close()

        print(f"→ Balken-Plot für Dataset '{dataset}' gespeichert als {output}")


def main():
    print("\n=== Rekursive JSON-Analyse startet ===\n")

    all_df = collect_all_results(ROOT_DIR)

    if all_df.empty:
        print("❌ Abbruch: keine Daten.")
        return

    plot_bar_charts(all_df, ROOT_DIR)

    print("\n=== Analyse abgeschlossen ===")


if __name__ == "__main__":
    main()
