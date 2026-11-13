# Thực nghiệm các thuật toán tách tiến – lùi

Repo chứa phần cài đặt và thực nghiệm cho đồ án về bài toán bao hàm biến phân và ứng dụng trong tối ưu danh mục đầu tư. Các thuật toán được viết bằng Python; phần mô tả ví dụ, thiết lập tham số và kết quả nằm trong các notebook.

## Nội dung repo

| File / thư mục | Nội dung |
|---|---|
| [FBM.py](FBM.py) | Thuật toán Forward–Backward (FBM). |
| [MFBSM_Tseng.py](MFBSM_Tseng.py) | Thuật toán Forward–Backward cải tiến của Tseng (MFBSM). |
| [RMFBSM.py](RMFBSM.py) | Thuật toán Forward–Backward có nới lỏng (RFBSM). |
| [IRFBSM.py](IRFBSM.py) | Thuật toán Forward–Backward có nới lỏng và quán tính (IRFBSM). |
| [common_libs.py](common_libs.py) | Các hàm dùng chung để kiểm tra vectơ, lập bảng kết quả và vẽ đồ thị hội tụ. |
| [VIP_full_algorythms.ipynb](VIP_full_algorythms.ipynb) | Ví dụ kiểm thử và so sánh bốn thuật toán trên cùng một bài toán. |
| [VIP_portfolio_optimize.ipynb](VIP_portfolio_optimize.ipynb) | Tiền xử lý dữ liệu, xây dựng mô hình tối ưu danh mục đầu tư và chạy thử bốn thuật toán trên FF10, FF48, FF100. |
| [Data/](Data/) | Ba bộ dữ liệu lợi suất tháng FF10, FF48 và FF100, tương ứng với 10, 48 và 100 danh mục. |

## Thực nghiệm kiểm tra thuật toán

`VIP_full_algorythms.ipynb`. 
Notebook trình bày một ví dụ có nghiệm chính xác là `(2, 3)` để kiểm tra khả năng chạy đúng của các thuật toán, sau đó chạy lần lượt FBM, MFBSM, RFBSM và IRFBSM. Mỗi thuật toán có bảng nghiệm tại các mốc lặp; cuối notebook là đồ thị so sánh độ chênh lệch giữa hai nghiệm liên tiếp. Phần cài đặt chi tiết nằm trong file `.py` tương ứng.

Các tham số được đặt ở đầu ví dụ, gồm điểm khởi tạo, cỡ bước, tham số nới lỏng, tham số quán tính, số vòng lặp tối đa và ngưỡng dừng.

## Thực nghiệm tối ưu danh mục đầu tư (portfolio)

Chạy thử **FBM, MFBSM của Tseng, RFBSM và IRFBSM** trong [VIP_portfolio_optimize.ipynb](VIP_portfolio_optimize.ipynb) với ba bộ dữ liệu lợi suất tháng **FF10, FF48 và FF100**, tương ứng với 10, 48 và 100 tài sản.

Lấy **60 tháng gần nhất** của mỗi bộ dữ liệu để chạy thử; khởi tạo bằng danh mục phân bổ đều `x0 = (1/n, ..., 1/n)` và lấy lợi suất kỳ vọng của danh mục này làm lợi suất mục tiêu.

### Cấu hình chạy thử

| Tham số | Giá trị |
|---|---|
| Cỡ bước ban đầu | `gamma = 0.8 / L`, với `L = (2/T) * ||R||₂²` |
| Tham số nới lỏng | `theta = 0.8` |
| Tham số cập nhật cỡ bước | `mu_step = 0.9` |
| Tham số quán tính | `alpha = 0.1` |
| Số vòng lặp tối đa | `max_iter = 100000` |
| Ngưỡng dừng | `tol = 1e-8` |

Tiêu chuẩn dừng dựa trên chuẩn Euclid của hiệu hai nghiệm liên tiếp: `||x^(k+1) - x^k||₂ <= tol`. 

### Kết quả chạy thử đã lưu trong notebook

| Bộ dữ liệu | FBM | MFBSM | RFBSM | IRFBSM |
|---|---:|---:|---:|---:|
| FF10 | 698 | 710 | 873 | 792 |
| FF48 | 3763 | 3771 | 4631 | 4203 |
| FF100 | 9530 | 9539 | 11630 | 10592 |
