import numpy as np
from common_libs import check_vector

def irfbsm(x_minus1, x0, B, prox, gamma, theta, mu, alpha,
           max_iter=10000, tol=1e-8,
           iter_list=(1, 10, 100, 1000, 10000)):
    """
    INERTIAL RELAXED FORWARD-BACKWARD SPLITTING METHOD (IRFBSM)

    Bài toán:
        0 ∈ A(x) + B(x)

    Công thức lặp:
        w^k = x^k + alpha*(x^k - x^(k-1))

        y^k = J_(gamma_k*A)(w^k - gamma_k*B(w^k))

        x^(k+1) = (1-theta)*w^k + theta*y^k
                  + theta*gamma_k*(B(w^k)-B(y^k))

    Cập nhật cỡ bước:
        gamma_(k+1) = min(
            gamma_k,
            mu*||w^k-y^k|| / ||B(w^k)-B(y^k)||
        )

        Nếu mẫu số bằng 0, giữ nguyên gamma_k.

    Residual tại vòng k:
        r_k = ||x^k - x^(k-1)||_2

    Tiêu chuẩn dừng:
        residual <= tol

    Input:
        x_minus1  : nghiệm khởi tạo x^(-1), shape (N,)
        x0        : nghiệm khởi tạo x^0, shape (N,)
        B         : toán tử đơn trị
        prox      : toán tử giải
        gamma     : cỡ bước khởi tạo
        theta     : tham số nới lỏng, 0 < theta <= 1
        mu        : tham số cập nhật cỡ bước, 0 < mu < 1
        alpha     : tham số quán tính, 0 <= alpha < 1
        max_iter  : số vòng lặp tối đa
        tol       : sai số cho phép
        iter_list : các vòng lặp cần lưu nghiệm

    Output:
        dict gồm nghiệm, lịch sử lặp, residual
        và lịch sử cỡ bước gamma.
    """

    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    x = check_vector(x, x.shape, "x0")

    x_prev = check_vector(x_minus1, x.shape, "x_minus1")

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma phải là số dương hữu hạn.")

    if not np.isfinite(theta) or not (0 < theta <= 1):
        raise ValueError("theta phải thuộc (0, 1].")

    if not np.isfinite(mu) or not (0 < mu < 1):
        raise ValueError("mu phải thuộc (0, 1).")

    if not np.isfinite(alpha) or not (0 <= alpha < 1):
        raise ValueError("alpha phải thuộc [0, 1).")

    condition = (
        theta * (1 - mu**2) / (2 - theta + mu * theta)**2
        + (1 - theta) / theta
    )

    inertial_bound = alpha * (1 + alpha) / (1 - alpha)**2

    if condition <= inertial_bound:
        raise ValueError("Các tham số alpha, theta, mu không thỏa điều kiện IRFBSM.")

    res = {0: x.copy()}
    residual_history = {0: np.nan}
    gamma_history = {0: gamma}

    converged = False
    residual = float("inf")
    iterations = 0

    for k in range(1, max_iter + 1):

        w = check_vector(x + alpha * (x - x_prev), x.shape, "w")

        Bw = check_vector(B(w), x.shape, "B(w)")
        y = check_vector(prox(w - gamma * Bw, gamma), x.shape, "y")

        By = check_vector(B(y), x.shape, "B(y)")
        x_new = check_vector(
            (1 - theta) * w + theta * y + theta * gamma * (Bw - By),
            x.shape,
            "x_new"
        )

        residual = float(np.linalg.norm(x_new - x, ord=2))
        residual_history[k] = residual

        numerator = float(np.linalg.norm(w - y, ord=2))
        denominator = float(np.linalg.norm(Bw - By, ord=2))

        if denominator > 0:
            gamma_new = min(gamma, mu * numerator / denominator)
        else:
            gamma_new = gamma

        if not np.isfinite(gamma_new) or gamma_new <= 0:
            raise ValueError("Cỡ bước gamma mới không hợp lệ.")

        x_prev = x.copy()
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