"""Các solver NumPy độc lập với dữ liệu và mô hình tài chính."""

import numpy as np


def fbm(x0, B, prox, gamma, max_iter=10000,
        tol=1e-8, iter_list=(1, 10, 100, 1000, 10000)):
    """Giải 0 ∈ A(x) + B(x) bằng Forward–Backward Method.

    Bước lặp: x_new = prox(x - gamma * B(x), gamma).

    Đầu vào
    -------
    x0 : array_like, shape (N,)
        Vector khởi tạo hữu hạn, N > 0; được sao chép sang float64.
    B : callable
        B(x) trả vector hữu hạn shape (N,).
    prox : callable
        prox(y, gamma) là resolvent J_(gamma*A)(y), trả shape (N,).
        Các callback không được sửa dữ liệu đầu vào tại chỗ.
    gamma : float > 0
        Cỡ bước cố định, hữu hạn, do nơi gọi lựa chọn.
    max_iter : int > 0
        Số vòng lặp tối đa.
    tol : float > 0
        Ngưỡng dừng hữu hạn cho sai khác bước lặp.
    iter_list : iterable[int]
        Các vòng lặp không âm cần lưu; luôn lưu vòng 0. Chỉ lưu các
        vòng thực sự đã chạy, không tiếp tục lặp sau khi đạt tol.

    Đầu ra
    ------
    dict với x (ndarray float64, shape (N,)), iterations (int),
    converged (bool), residual (float), res (dict[int, ndarray]) và
    stop_reason ('tolerance' hoặc 'max_iter'). Các snapshot độc lập.
    residual = ||x_k - x_(k-1)||_2 tại bước cuối; converged chỉ cho biết
    residual <= tol, không phải chứng nhận sai số tới nghiệm chính xác.

    Điều kiện hội tụ chuẩn: A đơn điệu cực đại, B beta-cocoercive
    (beta > 0), tập nghiệm khác rỗng và 0 < gamma < 2*beta. Nếu B là
    gradient của hàm lồi L-trơn với L > 0, có thể dùng 0 < gamma < 2/L.
    Chỉ tính đơn điệu và Lipschitz của B chưa đủ cho FBM nói chung.
    Solver không tự suy ra hoặc kiểm tra các giả thiết toán học này.

    Ví dụ: B(x)=x, A=0, prox(y, gamma)=y; nghiệm là vector 0.
    >>> result = fbm(np.array([2., -1.]), lambda x: x,
    ...              lambda y, gamma: y, gamma=1.)
    >>> result['converged']
    True
    """
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError('x0 phải là vector hữu hạn shape (N,), N > 0')
    if not callable(B) or not callable(prox):
        raise TypeError('B và prox phải là các hàm callable')
    if (not np.isscalar(gamma) or not np.isrealobj(gamma)
            or not np.isfinite(gamma) or gamma <= 0):
        raise ValueError('gamma phải là số thực hữu hạn > 0')
    if (not np.isscalar(tol) or not np.isrealobj(tol)
            or not np.isfinite(tol) or tol <= 0):
        raise ValueError('tol phải là số thực hữu hạn > 0')
    if (isinstance(max_iter, (bool, np.bool_))
            or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0):
        raise ValueError('max_iter phải là số nguyên > 0')
    iter_list = tuple(iter_list)
    if any(isinstance(k, (bool, np.bool_))
           or not isinstance(k, (int, np.integer)) or k < 0
           for k in iter_list):
        raise ValueError('iter_list phải chứa các số nguyên không âm')
    iter_list = set(iter_list)
    gamma = float(gamma)
    tol = float(tol)

    res = {0: x.copy()}
    converged = False
    residual = float('inf')
    iterations = 0
    stop_reason = 'max_iter'

    for k in range(1, max_iter + 1):
        Bx = np.asarray(B(x), dtype=float)
        if Bx.shape != x.shape or not np.isfinite(Bx).all():
            raise ValueError('B(x) phải là vector hữu hạn cùng shape với x0')
        y = x - gamma * Bx
        if not np.isfinite(y).all():
            raise ValueError('Bước tiến tạo giá trị NaN/Inf')
        x_new = np.asarray(prox(y, gamma), dtype=float)
        if x_new.shape != x.shape or not np.isfinite(x_new).all():
            raise ValueError('prox phải trả vector hữu hạn cùng shape với x0')
        residual = float(np.linalg.norm(x_new - x))
        if not np.isfinite(residual):
            raise ValueError('Sai khác bước lặp không hữu hạn')
        x = x_new.copy()
        iterations = k
        if k in iter_list:
            res[k] = x.copy()
        if residual <= tol:
            converged = True
            stop_reason = 'tolerance'
            break

    return {
        'x': x,
        'iterations': iterations,
        'converged': converged,
        'residual': residual,
        'res': res,
        'stop_reason': stop_reason,
    }
