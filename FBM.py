import numpy as np
import pandas as pd


def fbm(x0, B, prox, gamma, max_iter=10000,
        tol=1e-8, iter_list=(1, 10, 100, 1000, 10000)):
    """
    FORWARD-BACKWARD METHOD (FBM)

    Bài toán:
        0 ∈ A(x) + B(x)

    Công thức lặp:
        x^(k+1) = J_(gamma*A)(x^k - gamma*B(x^k))

    Residual tại vòng k:
        r_k = ||x^k - x^(k-1)||_2

    Tiêu chuẩn dừng:
        residual <= tol

    Input:
        x0        : nghiệm khởi tạo, shape (N,)
        B         : toán tử đơn trị
        prox      : toán tử giải J_(gamma*A)
        gamma     : cỡ bước
        max_iter  : số vòng lặp tối đa
        tol       : ngưỡng dừng
        iter_list : các vòng muốn lưu nghiệm

    Output:
        dict gồm nghiệm, lịch sử lặp và residual.
    """

    x = np.asarray(x0, dtype=float).reshape(-1).copy()

    if not np.isfinite(x).all():
        raise ValueError("x0 có NaN hoặc Inf.")

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma phải là số dương hữu hạn.")

    res = {0: x.copy()}
    residual_history = {0: np.nan}

    converged = False
    residual = float("inf")
    iterations = 0

    for k in range(1, max_iter + 1):

        # Step 1: Tính B(x^(k-1))
        step1 = np.asarray(
            B(x), dtype=float
        ).reshape(-1)

        # Step 2: Forward
        step2 = x - gamma * step1

        # Step 3: Backward
        x_new = np.asarray(
            prox(step2, gamma),
            dtype=float
        ).reshape(-1)

        if x_new.shape != x.shape:
            raise ValueError("Sai kích thước nghiệm.")

        if not np.isfinite(x_new).all():
            raise ValueError("Nghiệm có NaN hoặc Inf.")

        # r_k = ||x^k - x^(k-1)||_2
        residual = float(
            np.linalg.norm(x_new - x, ord=2)
        )

        residual_history[k] = residual

        # Cập nhật và lưu x^k
        x = x_new.copy()
        iterations = k

        if k in iter_list:
            res[k] = x.copy() #mảng nghiệm qua từng vòng lặp

        # điều kiện dừng
        if residual <= tol:
            converged = True
            break

    # nghiệm cuối
    res[iterations] = x.copy()

    return {
        "x": x.copy(),
        "iterations": iterations,
        "converged": converged,
        "residual": residual,
        "res": res,
        "residual_history": residual_history
    }
