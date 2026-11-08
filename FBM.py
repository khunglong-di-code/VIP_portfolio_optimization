def fbm(x0, B, prox, gamma, max_iter=10000,
        tol=1e-8, iter_list=(1, 10, 100, 1000, 10000)):
    """
    Forward-Backward

    Giải: 0 ∈ A(x) + B(x)

    x0        : nghiệm khởi tạo, shape (N,)
    B         : toán tử đơn trị B(x)
    prox      : toán tử giải J_(gamma*A)
    gamma     : cỡ bước
    max_iter  : số vòng lặp tối đa
    tol       : sai số 2 lần lặp
    iter_list : các vòng lặp cần lưu nghiệm

    Output:
        dict gồm x, iterations, converged,
        residual, res
    """
    x = np.asarray(x0, dtype=float).reshape(-1).copy()

    if gamma <= 0:
        raise ValueError("gamma phải dương.")

    res = {0: x.copy()}
    converged = False
    residual = float("inf")
    iterations = 0

    for k in range(1, max_iter + 1):

        step1 = B(x)
        step2 = x - gamma * step1
        x_new = np.asarray(
            prox(step2, gamma),
            dtype=float
        ).reshape(-1)

        if x_new.shape != x.shape:
            raise ValueError("Sai kích thước nghiệm.")

        if not np.isfinite(x_new).all():
            raise ValueError("Nghiệm có NaN hoặc Inf.")

        residual = float(np.linalg.norm(x_new - x))

        x = x_new.copy()
        iterations = k

        if k in iter_list:
            res[k] = x.copy()

        if residual <= tol:
            converged = True
            break

    return {
        "x": x,
        "iterations": iterations,
        "converged": converged,
        "residual": residual,
        "res": res
    }

