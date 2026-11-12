import numpy as np
import pandas as pd
from common_libs import check_vector, prox

def rfbsm(x0, B, prox, gamma, theta, mu, max_iter=10000,
          tol=1e-8, iter_list=(1, 10, 100, 1000, 10000)):
    """
    RELAXED FORWARD-BACKWARD SPLITTING METHOD (RFBSM)

    Bài toán:
        0 ∈ A(x) + B(x)

    Công thức lặp:
        y^k = J_(gamma_k*A)(x^k - gamma_k*B(x^k))

        x^(k+1) = (1-theta)*x^k + theta*y^k
                  + theta*gamma_k*(B(x^k)-B(y^k))

    Cập nhật cỡ bước:
        gamma_(k+1) = min(
            gamma_k,
            mu*||x^k-y^k|| / ||B(x^k)-B(y^k)||
        )

        Nếu mẫu số bằng 0, giữ nguyên gamma_k.

    Residual tại vòng k:
        r_k = ||x^k - x^(k-1)||_2

    Tiêu chuẩn dừng:
        residual <= tol

    Input:
        x0        : nghiệm khởi tạo, shape (N,)
        B         : toán tử đơn trị
        prox      : toán tử giải
        gamma     : cỡ bước khởi tạo
        theta     : tham số nới lỏng, 0 < theta <= 1
        mu        : tham số cập nhật cỡ bước, 0 < mu < 1
        max_iter  : số vòng lặp tối đa
        tol       : sai số cho phép
        iter_list : các vòng lặp cần lưu nghiệm

    Output:
        dict gồm nghiệm, lịch sử lặp, residual
        và lịch sử cỡ bước gamma.
    """
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    x = check_vector(x, x.shape, "x0")

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma phải là số dương hữu hạn.")

    if not np.isfinite(theta) or not (0 < theta <= 1):
        raise ValueError("theta phải thuộc (0, 1].")

    if not np.isfinite(mu) or not (0 < mu < 1):
        raise ValueError("mu phải thuộc (0, 1).")

    res = {0: x.copy()}
    residual_history = {0: np.nan}
    gamma_history = {0: gamma}

    converged = False
    residual = float("inf")
    iterations = 0

    for k in range(1, max_iter + 1):

        Bx = check_vector(B(x), x.shape, "B(x)")
        y = check_vector(prox(x - gamma * Bx, gamma), x.shape, "y")

        By = check_vector(B(y), x.shape, "B(y)")
        x_new = check_vector(
            (1 - theta) * x + theta * y + theta * gamma * (Bx - By),
            x.shape,
            "x_new"
        )

        residual = float(np.linalg.norm(x_new - x, ord=2))
        residual_history[k] = residual

        numerator = float(np.linalg.norm(x - y, ord=2))
        denominator = float(np.linalg.norm(Bx - By, ord=2))

        if denominator > 0:
            gamma_new = min(gamma, mu * numerator / denominator)
        else:
            gamma_new = gamma

        if not np.isfinite(gamma_new) or gamma_new <= 0:
            raise ValueError("Cỡ bước gamma mới không hợp lệ.")

        x = x_new.copy()
        gamma = gamma_new
        iterations = k

        gamma_history[k] = gamma

        if k in iter_list:
            res[k] = x.copy()

        if residual <= tol:
            converged = True
            break

    res[iterations] = x.copy()

    return {
        "x": x.copy(),
        "iterations": iterations,
        "converged": converged,
        "residual": residual,
        "res": res,
        "residual_history": residual_history,
        "gamma_history": gamma_history
    }
