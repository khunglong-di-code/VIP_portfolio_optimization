# Hướng dẫn sử dụng

Thư mục chứa mã thực nghiệm về bài toán bao hàm biến phân (VIP), thuật toán Forward–Backward (FBM) và dữ liệu phục vụ bài toán tối ưu danh mục đầu tư.

## Các file chính

| File | Nội dung |
|---|---|
| `FBM.py` | Hàm `fbm()` thực hiện thuật toán Forward–Backward. |
| `VIP_full_algorythms.ipynb` | Giải thích thuật toán và ví dụ chạy FBM, hiển thị bảng kết quả lặp. |
| `VIP_portfolio_optimize.ipynb` | Đọc, làm sạch dữ liệu và tạo cửa sổ huấn luyện/kiểm thử cho danh mục đầu tư. Hiện mới có phần chuẩn bị dữ liệu. |
| `FF10_...xlsx`, `FF48_...xlsx`, `FF100_...xlsx` | Dữ liệu lợi suất tháng của 10, 48 và 100 danh mục. |

## Cài đặt lần đầu

Cài Python, mở terminal PowerShell trong thư mục này rồi chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib scipy openpyxl jupyterlab ipykernel
```

Mỗi lần muốn mở notebook, chạy:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

JupyterLab sẽ mở trên trình duyệt. File `.ipynb` là notebook, gồm các ô (cell) chứa phần giải thích hoặc mã Python.

## Cách chạy

1. **Thử thuật toán:** mở `VIP_full_algorythms.ipynb`, chọn **Run → Run All Cells** để chạy từ trên xuống. Có thể chạy từng cell bằng **Shift + Enter**. Kết quả là bảng nghiệm theo vòng lặp và độ chênh lệch giữa hai nghiệm liên tiếp (residual).
2. **Chuẩn bị dữ liệu danh mục:** mở `VIP_portfolio_optimize.ipynb`. Trước khi chạy, thay ba đường dẫn `PATH_FF10`, `PATH_FF48`, `PATH_FF100` bằng tên file tương đối như sau:

   ```python
   PATH_FF10 = "FF10_AVG_Value_Weighted_Returns_Monthly.xlsx"
   PATH_FF48 = "FF48_AVG_Value_Weighted_Returns_Monthly.xlsx"
   PATH_FF100 = "FF100_AVG_Value_Weighted_Returns_Monthly.xlsx"
   ```

   Sau đó chạy các cell từ trên xuống. Kết quả nằm trong `windows_ff10`, `windows_ff48`, `windows_ff100`, chứa dữ liệu huấn luyện, kiểm thử và các tham số đầu vào.

## Tham số thường chỉnh

- **FBM:** `x0` là nghiệm khởi tạo; `B` và `prox` mô tả bài toán; `gamma` là cỡ bước cần chọn theo điều kiện thuật toán; `max_iter` là số vòng lặp tối đa; `tol` là ngưỡng dừng; `iter_list` là các vòng muốn lưu nghiệm để xem bảng.
- **Dữ liệu:** `CLEAN=True`, `FILL="zero"` thay dữ liệu thiếu bằng 0. Mặc định dùng 60 tháng huấn luyện, 12 tháng kiểm thử, trong giai đoạn kiểm thử 07/1976–06/2006; chỉnh qua `TRAIN_MONTHS`, `TEST_MONTHS`, `FIRST_TEST`, `LAST_TEST`.

## Khi gặp lỗi

- **Không tìm thấy `FBM` hoặc file Excel:** mở JupyterLab từ đúng thư mục này và giữ các file ở cùng thư mục với notebook.
- **Thiếu thư viện:** chạy lại lệnh cài đặt ở trên và chọn kernel dùng môi trường `.venv`.
- **Sau khi sửa `FBM.py`, kết quả vẫn cũ hoặc báo `KeyError: 'residual_history'`:** chọn **Kernel → Restart Kernel and Run All Cells** để nạp lại hàm và tạo lại `result`.
