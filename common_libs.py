import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def check_vector(v, shape, name="Nghiệm"):
    """
    Kiểm tra vectơ:
        1. Đúng kích thước
        2. Không chứa NaN hoặc Inf

    Input:
        v     : vectơ cần kiểm tra
        shape : kích thước yêu cầu
        name  : tên vectơ để thông báo lỗi

    Output:
        Vectơ NumPy một chiều hợp lệ.
    """

    v = np.asarray(v, dtype=float).reshape(-1)

    if v.shape != shape:
        raise ValueError(f"{name} sai kích thước.")

    if not np.isfinite(v).all():
        raise ValueError(f"{name} có NaN hoặc Inf.")

    return v

def show_iteration_table(result, decimals=8):

    rows = []

    has_gamma = "gamma_history" in result

    for k, x in sorted(result["res"].items()):

        vector_str = np.array2string(
            x,
            precision=decimals,
            separator=", ",
            suppress_small=False,
            threshold=np.inf,
            max_line_width=100000
        )

        row = {
            "Iteration (k)": k,
            "x^k (vector)": vector_str,
            "Residual ||x^k - x^(k-1)||₂":
                result["residual_history"].get(k, np.nan)
        }

        if has_gamma:
            row["Gamma"] = result["gamma_history"].get(k, np.nan)

        row["Final"] = k == result["iterations"]

        rows.append(row)

    table = pd.DataFrame(rows)

    table["Residual ||x^k - x^(k-1)||₂"] = (
        table["Residual ||x^k - x^(k-1)||₂"]
        .map(
            lambda v: f"{v:.3e}"
            if pd.notna(v) else "—"
        )
    )

    if has_gamma:
        table["Gamma"] = table["Gamma"].map(
            lambda v: f"{v:.6e}"
            if pd.notna(v) else "—"
        )

    return table

def plot_convergence(results, tol=1e-8):

    plt.figure(figsize=(9, 5))

    for name, result in results.items():

        history = result["residual_history"]

        iterations = sorted(
            k for k in history
            if k > 0 and np.isfinite(history[k]) and history[k] > 0
        )

        residuals = [history[k] for k in iterations]

        plt.plot(
            iterations,
            residuals,
            linewidth=1.5,
            label=name
        )

    plt.axhline(
        y=tol,
        color="gray",
        linestyle="--",
        label=f"Tolerance ({tol:.0e})"
    )

    plt.yscale("log")

    plt.xlabel("Iteration (k)")
    plt.ylabel(r"$\|x^k-x^{k-1}\|_2$")
    plt.title("Convergence Diagnostics")

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
