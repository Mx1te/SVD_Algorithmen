import numpy as np


def load_matrix(path: str) -> np.ndarray:
    """
    Liest eine Matrix aus einer Textdatei ein.
    Erwartet eine Datei mit Leerzeichen-getrennten Zahlen.
    """
    A = np.loadtxt(path)
    return A


def compute_svd(A: np.ndarray):
    """
    Berechnet die Singulärwertzerlegung A = U * Σ * V^T.
    full_matrices=False erzeugt die reduzierte SVD.
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U, S, Vt


def reconstruct_rank_k(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int):
    """
    Rekonstruiert die Rang-k-Approximation einer Matrix.
    A_k = Summe über i=1..k von (S[i] * U[:,i] * Vt[i,:])
    """
    m, n = U.shape[0], Vt.shape[1]
    Ak = np.zeros((m, n))

    for i in range(k):
        Ak += S[i] * np.outer(U[:, i], Vt[i, :])

    return Ak


def save_reconstruction(Ak: np.ndarray, filename: str):
    """
    Speichert eine rekonstruierte Matrix in eine Textdatei.
    """
    np.savetxt(filename, Ak, fmt="%.3f")


def storage_savings(m: int, n: int, k: int):
    """
    Berechnet die Speicherersparnis gegenüber der Originalmatrix.

    Originalspeicher = m * n
    SVD-Speicher = U(m,k) + S(k) + V(k,n) = mk + k + nk = k*(m + n + 1)
    """
    original = m * n
    svd = k * (m + n + 1)
    savings = 1 - svd / original
    return original, svd, savings


def main():
    # -------------------------------------------
    # Aufgabe 4: Einlesen der Matrix aus Datei
    # -------------------------------------------
    A = load_matrix("SVD_F.txt")
    print("Matrix eingelesen:", A.shape)

    # -------------------------------------------
    # Aufgabe 3: SVD-Berechnung
    # -------------------------------------------
    U, S, Vt = compute_svd(A)
    print("SVD berechnet:")
    print("U:", U.shape, "  S:", S.shape, "  Vt:", Vt.shape)

    # -------------------------------------------
    # Aufgabe 5: Rekonstruktion für verschiedene k
    # -------------------------------------------
    k_values = [1, 2, 3, 5, 8, 10, len(S)]

    for k in k_values:
        Ak = reconstruct_rank_k(U, S, Vt, k)
        filename = f"Rekonstruktion_k_{k}.txt"
        save_reconstruction(Ak, filename)
        print(f"Rang-{k}-Rekonstruktion gespeichert:", filename)

    # -------------------------------------------
    # Aufgabe 6: Speicherersparnis berechnen
    # -------------------------------------------
    m, n = A.shape
    print("\nSpeicherberechnungen:")
    for k in [1, 3, 5, 8]:
        orig, svd, sav = storage_savings(m, n, k)
        print(f"k = {k}: Original={orig}, SVD={svd}, Ersparnis={sav*100:.1f}%")



if __name__ == "__main__":
    main()
