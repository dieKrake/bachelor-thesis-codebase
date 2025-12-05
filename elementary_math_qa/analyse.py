import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path("elementary_math_qa/zero_shot_results")

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


def plot_summary(all_results: pd.DataFrame):
    """
    Erstellt ein Box-and-Whisker-Diagramm für die Accuracy pro N.
    - Box: 1. und 3. Quartil
    - Linie in der Box: Median
    - Whisker: Standard (1.5 * IQR); Ausreißer als einzelne Punkte
    - y-Achse beginnt bei 0 und geht bis 1 (Accuracy)
    """

    # N-Werte sortieren
    Ns_sorted = sorted(all_results["N"].unique())

    # Für jedes N alle Accuracy-Werte sammeln
    data_per_N = [
        all_results.loc[all_results["N"] == n, "accuracy"].values
        for n in Ns_sorted
    ]

    plt.figure(figsize=(8, 6))

    # Boxplot erstellen
    bp = plt.boxplot(
        data_per_N,
        positions=range(1, len(Ns_sorted) + 1),
        widths=0.6,
        patch_artist=True,   # erlaubt gefüllte Boxen
        showmeans=False      # Mittelwert wird nicht explizit markiert
    )

    # dezentes Grau für die Boxen (wissenschaftlicher Look)
    for box in bp['boxes']:
        box.set(facecolor="#d9d9d9", edgecolor="black")

    # Achsenbeschriftung mit N als Ticks
    plt.xticks(range(1, len(Ns_sorted) + 1), Ns_sorted)
    plt.xlabel("Datensatzgröße N")
    plt.ylabel("Accuracy")
    plt.title("Zero-Shot-Accuracy über 10 zufällig generierte Datensätze aus elementary_math_qa")

    # y-Achse bei 0 starten lassen, Accuracy typischerweise bis 1
    plt.ylim(bottom=0.0, top=1.0)

    # horizontales Raster für bessere Ablesbarkeit
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    output = BASE_DIR / "overall_boxplot.png"
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    print(f"→ Boxplot gespeichert als {output}")


def main():
    print("\n=== Zero-Shot Analyse startet ===\n")

    all_results = []

    for folder in BASE_DIR.iterdir():
        if folder.is_dir():
            df = analyze_folder(folder)
            all_results.append(df)

    if not all_results:
        print("❌ Keine Unterordner gefunden.")
        return

    all_df = pd.concat(all_results, ignore_index=True)

    # Summary (min/max/mean) weiterhin für die Tabelle erzeugen
    summary = build_overall_summary(all_df)

    # NEU: Box-and-Whisker-Plot auf Basis aller Einzelwerte
    plot_summary(all_df)

    print("\n=== Analyse abgeschlossen ===")


if __name__ == "__main__":
    main()
