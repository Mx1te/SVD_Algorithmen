# SVD Algorithmen - Bildkompression mit Singulärwertzerlegung

Dieses Projekt demonstriert die Anwendung der Singulärwertzerlegung (SVD) zur Bildkompression und Matrix-Approximation.

🔗 **Repository:** [https://github.com/Mx1te/SVD_Algorithmen](https://github.com/Mx1te/SVD_Algorithmen)

## Übersicht

Das Projekt enthält Implementierungen zur:
- Berechnung der Singulärwertzerlegung (SVD) einer Matrix
- Rekonstruktion von Rang-k-Approximationen
- Visualisierung von Bildkompression mit verschiedenen k-Werten
- Analyse der Speicherersparnis und Rekonstruktionsfehler
- Berechnung des Energieerhalts


## Anforderungen

### Python-Version
- Python 3.8 oder höher

### Benötigte Pakete
```
numpy
matplotlib
```

## Installation

> **Hinweis:** Falls die ZIP-Datei nicht funktioniert, kann das Projekt direkt vom GitHub-Repository geklont werden.

1. **Repository klonen:**
   ```bash
   git clone https://github.com/Mx1te/SVD_Algorithmen.git
   cd SVD_Algorithmen
   ```

   **Alternative:** ZIP-Datei herunterladen und entpacken.

2. **Benötigte Pakete installieren:**
   ```bash
   pip install numpy matplotlib
   ```

## Verwendung

### Grundlegende Ausführung

Das Skript `svd_image_compression.py` mit Standardeinstellungen ausführen:

```bash
cd Programm
python svd_image_compression.py
```

Wichtig! `SVD_F.txt` muss in der Ordnerstruktur in der gleichen Ebene sein, sonst `--input <pfad>`

Dies führt die SVD-Bildkompression mit folgenden Standardwerten aus:
- Eingabedatei: `SVD_F.txt`
- Ausgabeordner: `svd_results`
- k-Werte: 1, 2, 3, 5, 8, 10

### Kommandozeilenargumente

Das Skript unterstützt folgende Argumente zur Anpassung:

#### `--input <pfad>`
Pfad zur Eingabedatei (Matrix-Datei).
```bash
python svd_image_compression.py --input meine_matrix.txt
```
**Standard:** `SVD_F.txt`

#### `--out <ordner>`
Ausgabeordner für alle Ergebnisse (PNG-Bilder, Textdateien, CSV).
```bash
python svd_image_compression.py --out ergebnisse
```
**Standard:** `svd_results`

#### `--ks <k-werte>`
Komma-getrennte Liste von k-Werten für die Rang-k-Approximationen.
```bash
python svd_image_compression.py --ks 1,3,5,10,15
```
**Standard:** `1,2,3,5,8,10`

#### `--no-text`
Verhindert das Speichern der rekonstruierten Matrizen als Textdateien (nur PNG-Bilder werden gespeichert).
```bash
python svd_image_compression.py --no-text
```

#### `--show`
Zeigt die Bilder interaktiv mit matplotlib an (nützlich für die Entwicklung).
```bash
python svd_image_compression.py --show
```

### Beispiele

**Beispiel 1:** Nur niedrige k-Werte ohne Textdateien
```bash
python svd_image_compression.py --ks 1,2,3 --no-text
```

**Beispiel 2:** Eigene Matrix mit vielen k-Werten
```bash
python svd_image_compression.py --input daten/matrix.txt --out meine_ergebnisse --ks 1,2,5,10,20,30
```

**Beispiel 3:** Interaktive Anzeige mit spezifischen k-Werten
```bash
python svd_image_compression.py --ks 1,5,10 --show
```

### Ausgabedateien

Nach der Ausführung finden Sie im Ausgabeordner:
- `original.png` - Visualisierung der Originalmatrix
- `reconstruction_k_*.png` - Visualisierungen der Rang-k-Approximationen
- `reconstruction_k_*.txt` - Rekonstruierte Matrizen als Textdateien (falls nicht `--no-text`)
- `results_summary.csv` - Zusammenfassung aller Metriken (Fehler, Energie, Speicherersparnis)

### Funktionen

Das Skript bietet:
- Visualisierung der Matrix als Grauwertbild
- Rekonstruktion für benutzerdefinierte k-Werte
- Berechnung von Fehlermaßen (Frobenius-Norm)
- Berechnung des Energieerhalts
- Berechnung der Speicherersparnis
- Automatische Erstellung von Visualisierungen und CSV-Berichten



## Lizenz

Dieses Projekt ist für akademische Zwecke erstellt.

## Hinweise

- Diese README wurde mit Unterstützung von GitHub Copilot (KI) erstellt.
- Bei Problemen mit der ZIP-Datei kann das Projekt direkt vom Repository geklont werden.

---

**Repository:** [https://github.com/Mx1te/SVD_Algorithmen](https://github.com/Mx1te/SVD_Algorithmen)
