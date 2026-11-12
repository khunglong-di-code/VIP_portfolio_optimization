import numpy as np
import pandas as pd
from common_libs import check_vector

def mfbm_tseng(x0, B, prox, gamma, max_iter=10000,
               tol=1e-8, iter_list=(1, 10, 100, 1000, 10000)):
    """
    MODIFIED FORWARD-BACKWARD METHOD (MFBSM) - TSENG

    Bài toán:
        0 ∈ A(x) + B(x)

    Công thức lặp:
        y^k     = J_(gamma*A)(x^k - gamma*B(x^k))
        x^(k+1) = y^k - gamma*(B(y^k) - B(x^k))

    Residual tại vòng k:
        r_k = ||x^k - x^(k-1)||_2

    Tiêu chuẩn dừng:
        residual <= tol

    Input:
        x0        : nghiệm khởi tạo, shape (N,)
        B         : toán tử đơn trị
        prox      : toán tử giải
        gamma     : cỡ bước cố định
        max_iter  : số vòng lặp tối đa
        tol       : sai số cho phép
        iter_list : các vòng lặp cần lưu nghiệm

    Output:
        dict gồm nghiệm, lịch sử lặp và residual.
    """

    x = np.asarray(x0, dtype=float).reshape(-1).copy()

    x = check_vector(x, x.shape, "x0")

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma phải là số dương hữu hạn.")

    res = {0: x.copy()}
    residual_history = {0: np.nan}

    converged = False
    residual = float("inf")
    iterations = 0

    for k in range(1, max_iter + 1):

        # Step 1: Tính B(x^(k-1))
        step1 = check_vector(
            B(x), x.shape, "B(x)"
        )

        # Step 2: Forward thứ nhất
        step2 = x - gamma * step1

        # Step 3: Backward
        y = check_vector(
            prox(step2, gamma),
            x.shape,
            "Nghiệm trung gian"
        )

        # Step 4: Tính B(y^(k-1))
        step3 = check_vector(
            B(y), x.shape, "B(y)"
        )

        # Step 5: Forward thứ hai
        x_new = check_vector(
            y - gamma * (step3 - step1),
            x.shape,
            "Nghiệm mới"
        )

        # r_k = ||x^k - x^(k-1)||_2
        residual = float(
            np.linalg.norm(x_new - x, ord=2)
        )

        residual_history[k] = residual

        # Cập nhật và lưu x^k
        x = x_new.copy()
        iterations = k

        if k in iter_list:
            res[k] = x.copy()

        # Điều kiện dừng
        if residual <= tol:
            converged = True
            break

    # Nghiệm cuối
    res[iterations] = x.copy()

    return {
        "x": x.copy(),
        "iterations": iterations,
        "converged": converged,
        "residual": residual,
        "res": res,
        "residual_history": residual_history
    }