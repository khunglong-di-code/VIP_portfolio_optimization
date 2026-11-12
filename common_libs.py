import numpy as np
import pandas as pd


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