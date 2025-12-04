import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path("navigate_200/zero_shot_results")

NS = ["40", "80", "120", "160"]


def load_accuracy_from_file(file_path: Path):
    """Lädt Accuracy aus JSON-Result-Datei."""
    data = json.loads(file_path.read_text(encoding="utf-8"))
    return data["result"]["accuracy"]


def analyze_folder(folder: Path):
    """
    Analysiert 1 Unterordner (z. B. navigate_200/zero_shot_results/23)
    und speichert eine eindeutige analysis-CSV im BASE_DIR:
    z.B. navigate_200/zero_shot_results/analysis_23.csv
    """

    print(f"Analysiere Ordner: {folder.name}")
    rows = []

    for n in NS:
        # suche JSON-Files die zu n passen
        candidates = list(folder.glob(f"zeroshot_eval_{n}_*.json"))
        if not candidates:
            print(f"⚠️ Kein File für n={n} in {folder.name}")
            continue

        # nimm jeweils das neueste file
        file = sorted(candidates)[-1]
        acc = load_accuracy_from_file(file)

        rows.append({
            "folder": folder.name,   # Ordnername (z.B. "23")
            "N": int(n),             # Datensatzgröße
            "accuracy": acc,
            "file": file.name        # konkrete JSON-Datei
        })

    df = pd.DataFrame(rows)

    # Eindeutiger Dateiname basierend auf Ordnernamen
    output_csv = BASE_DIR / f"analysis_{folder.name}.csv"
    df.to_csv(output_csv, index=False)
    print(f"→ gespeichert: {output_csv}")

    return df


def build_overall_summary(all_results: pd.DataFrame):
    """Erstellt Min–Max–Mean Tabelle für alle N (über alle Ordner)."""

    summary = (
        all_results
        .groupby("N")["accuracy"]
        .agg(["min", "max", "mean"])
        .reset_index()
    )

    summary.to_csv(BASE_DIR / "overall_summary.csv", index=False)
    print(f"→ Overall Summary gespeichert unter overall_summary.csv")

    return summary


def plot_summary(summary_df: pd.DataFrame):
    """Erstellt eine Min/Max/Mean Grafik für Accuracy pro N."""

    plt.figure(figsize=(10, 6))

    Ns = summary_df["N"]
    mean = summary_df["mean"]
    minv = summary_df["min"]
    maxv = summary_df["max"]

    # Bereich (min–max)
    plt.fill_between(Ns, minv, maxv, alpha=0.2, label="Min–Max Range")

    # Mittelwert
    plt.plot(Ns, mean, marker="o", linewidth=3, label="Mean Accuracy")

    plt.xlabel("Dataset Size (N)")
    plt.ylabel("Accuracy")
    plt.title("Zero-Shot Accuracy Summary Across All Test Folders")
    plt.grid(True)
    plt.legend()

    output = BASE_DIR / "overall_plot.png"
    plt.savefig(output)
    print(f"→ Grafik gespeichert als {output}")


def main():
    print("\n=== Zero-Shot Analyse startet ===\n")

    all_results = []

    for folder in BASE_DIR.iterdir():
        if folder.is_dir():
            df = analyze_folder(folder)
            all_results.append(df)

    # zu einem großen DF zusammenführen
    if not all_results:
        print("❌ Keine Unterordner gefunden.")
        return

    all_df = pd.concat(all_results, ignore_index=True)

    summary = build_overall_summary(all_df)
    plot_summary(summary)

    print("\n=== Analyse abgeschlossen ===")


if __name__ == "__main__":
    main()
