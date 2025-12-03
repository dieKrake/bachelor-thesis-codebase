Hier ist dein kompletter Text sauber in **Markdown** formatiert:

---

# Erklärung des Skripts und wie es PE2 aus dem Paper umsetzt

Ich gehe das Skript einmal von oben nach unten durch **und** erkläre parallel, wie das alles zur PE2-Beschreibung im Paper passt.

---

## 1. Grundidee

Dieses Skript ist eine „saubere“ Umsetzung von PE2:

- du hast **einen großen Datensatz D** (z. B. `navigate_full.json`)
- dieser wird in **drei Teile** gesplittet:

  - **T_train** – um Fehlbeispiele zu sammeln
  - **T_dev** – um zu entscheiden, welcher Prompt der beste ist
  - **T_test** – bleibt unangetastet, bis ganz am Ende zur finalen Messung

- PE2 sucht dann iterativ einen immer besseren Prompt.

---

## 2. Datentypen

```python
@dataclass
class Example:
    input: str
    label: str
```

Ein einzelner Datenpunkt: Eingabetext + korrektes Label.

```python
@dataclass
class PromptCandidate:
    prompt: str
    accuracy: float = 0.0  # Dev-Accuracy
    eval_results_dev: List[Dict[str, Any]] = ...
    eval_results_train: List[Dict[str, Any]] = ...
```

Ein **Prompt-Kandidat**, bestehend aus:

- `prompt`: der eigentliche Prompt
- `accuracy`: **Dev-Accuracy**, also wie gut der Prompt auf dem Dev-Set ist
- `eval_results_train`: Output- und Fehlerdaten auf T_train
- `eval_results_dev`: Evaluation auf T_dev

➡️ Im Paper entspricht das einem „Beam“–Knoten im Suchbaum.

---

## 3. Mtask = Prompt auf Dataset evaluieren

```python
def evaluate_prompt_on_dataset(...)
```

- bekommt Prompt + Dataset
- ruft das LLM für jedes Beispiel auf
- vergleicht Antwort mit Label
- erzeugt Records wie:

```json
{
  "Index": idx,
  "Input": "...",
  "correctAnswer": "...",
  "MtaskAnswer": "...",
  "judgement": true/false
}
```

→ gibt `(accuracy, eval_results)` zurück.

Im Paper: **M_task**, das Modell zur Bewertung eines Prompts.

---

## 4. run_mtask_for_candidate (wichtig für PE2)

```python
def run_mtask_for_candidate(...)
```

Mtask wird hier **zweimal** ausgeführt:

1. **auf T_dev** → bestimmt die Prompt-Qualität
2. **auf T_train** → liefert Fehlbeispiele für Mproposal

➡️ Genau wie im Paper:
T_train = Grundlage für Verbesserungen
T_dev = Auswahl des besten Prompts

---

## 5. Fehlerbatches B für Mproposal

```python
def build_failure_examples_text(...)
```

- sammelt alle Fehleinschätzungen aus T_train
- sampelt max. `failure_batch_size`
- formatiert sie als großen Textblock `examples_text`

Im Paper: Das ist genau das **Batch B**.

---

## 6. Mproposal = Prompt verbessern

```python
def run_mproposal(...)
```

- baut einen Meta-Prompt bestehen aus:

  - aktuellem Prompt
  - Fehlbeispielen B
  - Instruktionen + JSON-Schema

- ruft das LLM auf
- erwartet ein JSON:

```json
{
  "improved_prompt": "...",
  "reasoning": "..."
}
```

→ liefert neuen Prompt zurück.

Im Paper: **M_proposal**.

---

## 7. PE2-Loop / Beam Search

```python
def pe2_optimize_prompt(...)
```

Hier passiert Algorithmus 1 aus dem Paper.

### Start (Iteration 0)

- initialer Prompt wird evaluiert
- Fehler + Dev-Accuracy werden gespeichert
- `current_candidates = [base_candidate]`
- `all_candidates = [base_candidate]`

### Hauptschleife

Für jede Iteration:

1. **Beam auswählen (Top n nach Dev-Accuracy)**

   ```python
   parents = all_candidates[:beam_width]
   ```

2. **Abbruch bei perfekter Dev-Accuracy**

3. **Für jeden Elternprompt m Verbesserungen generieren**
   → mit Mproposal

4. **Duplikate vermeiden**

5. **Neuen Prompt auf Train+Dev evaluieren**

6. **Besten Prompt aktualisieren**

7. **Backtracking**
   Nicht nur letzte Runde, sondern **alle bisherigen Prompts** werden als Kandidaten genutzt
   → entspricht dem Paper.

---

## 8. Dataset laden & splitten

```python
train_set, dev_set, test_set = split_dataset(...)
```

- `T_train`: zum Fehler sammeln
- `T_dev`: zur Prompt-Auswahl
- `T_test`: am Ende für finale Messung

Das entspricht exakt dem Paper:
Train → Proposal-Generierung
Dev → Model Selection
Test → Final Evaluation

---

## 9. Finale Evaluation auf T_test

Nach der Optimierung wird:

```python
test_accuracy, test_results = evaluate_prompt_on_dataset(best_prompt, test_set)
```

→ T_test wurde vorher **nie benutzt**, also echte Generalisierung.

---

## 10. Warum vorher nur ein Dataset?

In deiner ursprünglichen Version:

- du hattest **nur einen** Datensatz
- hast ihn zur Optimierung und zur finalen Bewertung genutzt

Damit gilt:

- **T_train = T_dev = T_test = D**
- du misst also, wie gut der Prompt auf denselben Daten performt, auf denen du ihn optimierst
  → **Overfitting**, keine echte Aussage über Generalisierungsfähigkeit.

Die neue Implementation trennt diese Rollen:

| Zweck                | Datensatz |
| -------------------- | --------- |
| Prompt verbessern    | T_train   |
| Prompt auswählen     | T_dev     |
| Endgültige Bewertung | T_test    |

So kannst du seriös messen, ob der optimierte Prompt **wirklich** besser ist.

---
