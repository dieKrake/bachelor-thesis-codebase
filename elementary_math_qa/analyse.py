import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

method_name_dir = "few_shot_results"   # Ordnername
method_name = "fewshot"                # Präfix der Result-Files
dataset_name = "elementary_math_qa"
BASE_DIR = Path(f"{dataset_name}/{method_name_dir}")


def load_metrics_from_file(file_path: Path):
    """
    Lädt Accuracy + Testgröße (N) aus einer Few-Shot-JSON-Datei.
    Erwartet das Schema aus deinem Few-Shot-Skript:
      "dataset": { "test_size": ..., ... }
      "result":  { "accuracy": ... }
    """
    data = json.loads(file_path.read_text(encoding="utf-8"))
    accuracy = data["result"]["accuracy"]
    # test_size aus dem JSON (robuster als aus dem Dateinamen)
    test_size = data["dataset"].get("test_size", None)
    return test_size, accuracy


def analyze_folder(folder: Path):
    """
    Analysiert 1 Unterordner (z. B. elementary_math_qa/few_shot_results/5)
    und speichert eine eindeutige analysis-CSV im BASE_DIR:
    z.B. elementary_math_qa/few_shot_results/analysis_5.csv
    """

    print(f"Analysiere Ordner: {folder.name}")
    rows = []

    # Alle Few-Shot-JSONs in diesem Split-Ordner einsammeln
    # z.B. fewshot_eval_split5_train10_test40_...json
    candidates = list(folder.glob(f"{method_name}_eval_*.json"))
    if not candidates:
        print(f"⚠️ Keine {method_name}-Files in {folder.name} gefunden")
    else:
        for file in sorted(candidates):
            N, acc = load_metrics_from_file(file)

            if N is None:
                print(f"  ⚠️ Keine test_size in {file.name}, überspringe")
                continue

            rows.append({
                "folder": folder.name,   # Ordnername (z.B. "5")
                "N": int(N),             # Test-Datensatzgröße
                "accuracy": acc,
                "file": file.name        # konkrete JSON-Datei
            })

    if not rows:
        print(f"⚠️ Keine verwertbaren Resultate in {folder.name} – leere CSV.")
        df = pd.DataFrame(columns=["folder", "N", "accuracy", "file"])
    else:
        df = pd.DataFrame(rows)

    # Eindeutiger Dateiname basierend auf Ordnernamen
    output_csv = BASE_DIR / f"analysis_{folder.name}.csv"
    df.to_csv(output_csv, index=False)
    print(f"→ gespeichert: {output_csv}")

    return df


def build_overall_summary(all_results: pd.DataFrame):
    """Erstellt Min–Max–Mean Tabelle für alle N (über alle Ordner)."""

    # Falls all_results leer ist, direkt abbrechen
    if all_results.empty:
        print("⚠️ all_results ist leer – kein Summary möglich.")
        return pd.DataFrame()

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

    if all_results.empty:
        print("⚠️ all_results ist leer – kein Plot möglich.")
        return

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
    plt.title(
        f"{method_name}-Accuracy über zufällig generierte Datensätze aus {dataset_name}"
    )

    # y-Achse bei 0 starten lassen, Accuracy typischerweise bis 1
    plt.ylim(bottom=0.0, top=1.0)

    # horizontales Raster für bessere Ablesbarkeit
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    output = BASE_DIR / "overall_boxplot.png"
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    print(f"→ Boxplot gespeichert als {output}")


def main():
    print("\n=== Analyse startet ===\n")

    all_results = []

    for folder in BASE_DIR.iterdir():
        if folder.is_dir():
            df = analyze_folder(folder)
            # nur anhängen, wenn die DataFrame überhaupt Zeilen hat
            if not df.empty:
                all_results.append(df)

    if not all_results:
        print("❌ Keine verwertbaren Resultate in Unterordnern gefunden.")
        return

    all_df = pd.concat(all_results, ignore_index=True)

    # Summary (min/max/mean) für Tabelle erzeugen
    summary = build_overall_summary(all_df)

    # Box-and-Whisker-Plot auf Basis aller Einzelwerte
    plot_summary(all_df)

    print("\n=== Analyse abgeschlossen ===")


if __name__ == "__main__":
    main()
