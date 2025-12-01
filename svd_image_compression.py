"""
SV D Bildkompression - Fertige Abgabe-Version
Datei: svd_image_compression.py
Sprache: Deutsch (Kommentare auf Deutsch)

Inhalt:
- Einlesen der Matrix SVD_F.txt (15x25) als Grauwert-Bild
- Berechnung der (reduzierten) SVD mit numpy
- Rekonstruktion für eine Liste von k-Werten
- Visualisierung (Anzeige + Speichern als PNG)
- Speichern der rekonstruierten Matrizen als Textdateien
- Berechnung von Fehlermaßen (Frobenius-Norm) und Energieerhalt
- Berechnung und Ausgabe der Speicherersparnis
- Erstellung einer kompakten Ergebnisdatei (CSV)

Anforderungen zur Abgabe beachten:
- Das Skript läuft unter Python 3 (getestet mit 3.8+). Benötigte Pakete: numpy, matplotlib
- Starten: python svd_image_compression.py

Hinweis: Kommentare auf Deutsch (wie in der Aufgabenstellung gefordert).
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


# ---------------------------
# Hilfsfunktionen
# ---------------------------

def load_matrix(path: str) -> np.ndarray:
    """
    Liest eine Matrix aus einer Textdatei ein.
    Erwartet eine Datei mit Leerzeichen-getrennten Zahlen.
    Liefert eine 2D-NumPy-Array (float).
    """
    A = np.loadtxt(path)
    return A


def compute_svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet die reduzierte SVD A = U * diag(S) * Vt.
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U, S, Vt


def reconstruct_rank_k(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """
    Rekonstruiert die Rang-k-Approximation mittels Matrixprodukt (schnell & klar):
    A_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
    """
    if k <= 0:
        raise ValueError("k muss >= 1 sein")
    k = min(k, len(S))
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def save_reconstruction(Ak: np.ndarray, filename: str):
    """
    Speichert eine rekonstruierte Matrix in eine Textdatei (3 Nachkommastellen).
    """
    np.savetxt(filename, Ak, fmt="%.3f")


def reconstruction_error(A: np.ndarray, Ak: np.ndarray) -> float:
    """
    Berechnet die Frobenius-Norm des Fehlers: ||A - Ak||_F
    """
    return float(np.linalg.norm(A - Ak, ord='fro'))


def energy_retained(S: np.ndarray, k: int) -> float:
    """
    Anteil der durch die ersten k Singulärwerte erklärten Energie (Quadratsumme).
    Gibt Wert in [0,1] zurück.
    """
    k = min(k, len(S))
    total = np.sum(S ** 2)
    if total == 0:
        return 0.0
    return np.sum(S[:k] ** 2) / total


def storage_savings(m: int, n: int, k: int) -> Tuple[int, int, float]:
    """
    Berechnet Originalspeicher (Anzahl Werte), SVD-Speicher und relative Ersparnis.
    Original = m * n
    SVD-Speicher (Anzahl Werte) = k*(m + n + 1)  (U(m,k) + S(k) + V(k,n))
    Rückgabe: (original, svd, savings_rel)
    """
    original = m * n
    svd = k * (m + n + 1)
    savings = 1 - svd / original
    return original, svd, savings


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_matrix(A: np.ndarray, title: Optional[str] = None, savepath: Optional[str] = None):
    """
    Zeigt eine Matrix als Graustufenbild und speichert optional als PNG.
    """
    plt.figure(figsize=(4, 6))
    plt.imshow(A, cmap='gray', aspect='equal')
    if title:
        plt.title(title)
    plt.axis('off')
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=150)
    plt.close()


# ---------------------------
# Hauptfunktionalität
# ---------------------------

def process_file(
    input_path: str,
    output_dir: str,
    k_values: List[int],
    save_text_recons: bool = True,
    show_plots: bool = False
):
    """
    Führt die komplette Pipeline aus:
    - Einlesen
    - SVD
    - Rekonstruktionen für k_values
    - Speichern von PNG + optional Textdateien
    - Ausgabe einer Zusammenfassung (CSV + report outline)
    """
    ensure_dir(output_dir)

    A = load_matrix(input_path)
    m, n = A.shape
    U, S, Vt = compute_svd(A)

    # Original als Bild speichern
    orig_png = os.path.join(output_dir, 'original.png')
    plot_matrix(A, title='Original', savepath=orig_png)

    results = []  # für CSV: k, error, energy, original, svd, savings

    for k in k_values:
        Ak = reconstruct_rank_k(U, S, Vt, k)

        # Dateinamen
        png_name = os.path.join(output_dir, f'reconstruction_k_{k}.png')
        txt_name = os.path.join(output_dir, f'reconstruction_k_{k}.txt')

        # Speichern
        plot_matrix(Ak, title=f'Rang-{k}', savepath=png_name)
        if save_text_recons:
            save_reconstruction(Ak, txt_name)

        # Fehler / Energie / Speicher
        err = reconstruction_error(A, Ak)
        energy = energy_retained(S, k)
        orig, svd, savings = storage_savings(m, n, k)

        results.append((k, err, energy, orig, svd, savings))

        # Optionale Anzeige
        if show_plots:
            # kleine Anzeige für interaktives Arbeiten
            plt.imshow(Ak, cmap='gray')
            plt.title(f'Rang-{k} (E={energy*100:.1f}%, Fehler={err:.3f})')
            plt.axis('off')
            plt.show()

    # CSV speichern
    import csv
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['k', 'frobenius_error', 'energy_retained', 'original_values', 'svd_values', 'savings_relative'])
        for row in results:
            writer.writerow(row)

    

    print(f"Ergebnisordner: {output_dir}")
    print(f"Originalbild: {orig_png}")
    print(f"CSV-Zusammenfassung: {csv_path}")
    print("Fertig. Du findest dort PNGs, Text-Rekonstruktionen und eine results_summary.csv.")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='SVD Bildkompression - Fertige Abgabe-Version')
    parser.add_argument('--input', type=str, default='SVD_F.txt', help='Pfad zur Eingabedatei (Matrix)')
    parser.add_argument('--out', type=str, default='svd_results', help='Ausgabeordner')
    parser.add_argument('--ks', type=str, default='1,2,3,5,8,10', help='Komma-getrennte Liste von k-Werten (z.B. "1,3,5,8,10")')
    parser.add_argument('--no-text', action='store_true', help='Textdateien der Rekonstruktionen nicht speichern')
    parser.add_argument('--show', action='store_true', help='Bilder interaktiv anzeigen (matplotlib.show)')
    return parser.parse_args()


def main():
    args = parse_args()
    k_values = [int(x) for x in args.ks.split(',') if x.strip()]

    process_file(
        input_path=args.input,
        output_dir=args.out,
        k_values=k_values,
        save_text_recons=not args.no_text,
        show_plots=args.show
    )


if __name__ == '__main__':
    main()
