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
| [VIP_portfolio_optimize.ipynb](VIP_portfolio_optimize.ipynb) | Chuẩn bị dữ liệu và trình bày mô hình tối ưu danh mục đầu tư. |
| [Data/](Data/) | Ba bộ dữ liệu lợi suất tháng FF10, FF48 và FF100, tương ứng với 10, 48 và 100 danh mục. |

## chạy thực nghiệm

`VIP_full_algorythms.ipynb`. 
Notebook trình bày một ví dụ có nghiệm chính xác là `(2, 3)` để kiểm tra khả năng chạy đúng của các thuật toán, sau đó chạy lần lượt FBM, MFBSM, RFBSM và IRFBSM. Mỗi thuật toán có bảng nghiệm tại các mốc lặp; cuối notebook là đồ thị so sánh độ chênh lệch giữa hai nghiệm liên tiếp. Phần cài đặt chi tiết nằm trong file `.py` tương ứng.

Các tham số được đặt ở đầu ví dụ, gồm điểm khởi tạo, cỡ bước, tham số nới lỏng, tham số quán tính, số vòng lặp tối đa và ngưỡng dừng.

`VIP_portfolio_optimize.ipynb` 
Notebook xử lý dữ liệu và chạy thuật toán cho bài toán tối ưu danh mục đầu tư với 3 bộ dữ liệu F10,F48,F100


