# ĐA2 — Quy ước lập trình thuật toán bao hàm biến phân

> **Đọc file này trước khi viết, sửa hoặc tích hợp bất kỳ thuật toán nào.** Đây là hợp đồng giao diện (coding contract) của dự án. Ưu tiên mã Python/NumPy đơn giản, dễ đọc và bám sát công thức toán học.

## 1. Mục tiêu và phạm vi

- Nghiên cứu, cài đặt và đối chiếu các thuật toán giải **bao hàm biến phân** dạng `0 ∈ A(x) + B(x)`, trước mắt là **FBM (Forward–Backward Method)** và các phương pháp quán tính.
- **Notebook thuật toán** (`FBM.ipynb`, `SA_IFB.ipynb`, …): kiểm chứng bằng ví dụ toán học độc lập, không đọc dữ liệu tài chính.
- **`algorithms.py`**: chứa các hàm đã kiểm chứng để import và tái sử dụng.
- **`VIP_full.ipynb`**: đọc/làm sạch FF10, FF48, FF100; xây dựng bài toán cụ thể từ dữ liệu; gọi các hàm trong `algorithms.py`; đánh giá kết quả.
- **Rolling Window là tùy chọn của phần đánh giá thực nghiệm**; không được buộc thuật toán FBM phụ thuộc vào Rolling Window.
- Mã phải đơn giản: `numpy`, các hàm `def`, vòng lặp `for`/`while`, biến trung gian rõ ràng. Không tự ý chuyển sang `class`, framework, tự động hóa quá mức hoặc thay cấu trúc dự án.

## 2. Quy tắc bắt buộc khi một agent viết code

1. **Đọc README trước.** Giữ nguyên tên biến và hợp đồng hàm đã quy định; nếu cần đổi, nêu rõ và cập nhật README cùng nơi gọi.
2. **Tách thuật toán khỏi bài toán.** Solver chỉ nhận `x0`, `B`, `prox` và tham số; không đọc CSV, không nhận `DataFrame`, không tự gọi `ff48`, `R`, `mu`, `Sigma` bên trong solver.
3. **Bám sát toán học** của thuật toán/bài báo được chọn; không tự đồng nhất các thuật toán khác nhau hoặc mặc định một công thức bước lặp cho mọi bài toán.
4. **Không tự thay đổi mô hình tối ưu**, ràng buộc, cách chọn `gamma`, tiêu chuẩn dừng hoặc quy tắc `lambda` để làm code “chạy được”. Đề xuất sửa đổi phải được ghi là phương án khác.
5. **Không làm biến đổi dữ liệu đầu vào tại chỗ.** Với mảng nghiệm, dùng `.copy()` khi cần lưu lịch sử.
6. **Không báo “hội tụ” chỉ vì hết `max_iter`.** Luôn trả `converged`, `iterations`, `residual` và lý do dừng.
7. **Bất kỳ code mới nào cũng kèm ví dụ gọi hàm** và ghi rõ đầu vào/đầu ra, shape, kiểu dữ liệu.
8. **Không dùng thông tin tương lai** khi ước lượng `mu`, `Sigma`, chọn tham số hoặc xử lý missing trong thí nghiệm ngoài mẫu.

## 3. Quy ước tên biến và kích thước

| Tên | Ý nghĩa | Kiểu / kích thước chuẩn |
|---|---|---|
| `x`, `x0`, `x1`, `x_new`, `x_star` | nghiệm/trọng số và các bước lặp | `np.ndarray`, shape `(N,)` |
| `N` | số chiều / số danh mục | `int` |
| `T` | số tháng quan sát | `int` |
| `k` | chỉ số vòng lặp | `int` |
| `A` | toán tử đa trị trong `0 ∈ A(x)+B(x)` | ký hiệu toán học; không bắt buộc biểu diễn trực tiếp |
| `B` | toán tử đơn trị | callable `B(x) -> ndarray (N,)` |
| `prox` | toán tử giải (resolvent) `J_{gamma A}` | callable `prox(y, gamma) -> ndarray (N,)` |
| `gamma` | cỡ bước của thuật toán | `float > 0` hoặc dãy khi thuật toán yêu cầu |
| `lam` | tham số điều chuẩn `lambda` của **bài toán** | `float >= 0`; tránh dùng `lambda` làm tên biến Python |
| `max_iter` | giới hạn số vòng lặp | `int > 0` |
| `tol` | ngưỡng dừng | `float > 0` |
| `iter_list` | vòng lặp cần lưu nghiệm | iterable của `int` |
| `res` | từ điển snapshot nghiệm | `dict[int, ndarray]` |
| `R` | ma trận lợi suất đã đổi sang thập phân | `np.ndarray`, shape `(T, N)` |
| `R_train`, `R_test` | lợi suất huấn luyện/kiểm thử | `(T_train,N)`, `(T_test,N)` |
| `mu` | vector lợi suất kỳ vọng mẫu | shape `(N,)` |
| `Sigma` | ma trận hiệp phương sai mẫu | shape `(N,N)` |
| `rho` | lợi suất mục tiêu của mô hình danh mục | `float` |
| `x_equal` | trọng số đều `1/N` | shape `(N,)` |
| `asset_names` | tên các danh mục | `list[str]`, dài `N` |
| `C` | tập ràng buộc | định nghĩa toán học; phép chiếu được đóng gói trong `prox` khi phù hợp |
| `L` | hằng số Lipschitz của `B` (nếu có) | `float >= 0` |

**Chuẩn NumPy:** vector dùng shape `(N,)`, **không dùng** `(N,1)` cho code mới. Tính với `dtype=float` (`float64`). Dùng `@` cho tích ma trận; `np.linalg.norm(x)` cho chuẩn Euclid. Với ma trận, `np.linalg.norm(F, ord=2)` là chuẩn phổ. Nếu cần bình phương chuẩn, viết rõ `np.linalg.norm(F, ord=2) ** 2`. **Không dùng `np.linalg.eig` để tự tính norm khi hàm chuẩn đã có.**

**Tránh trùng nghĩa:** `mu` là lợi suất kỳ vọng **trong notebook tài chính**; nếu công thức thuật toán có một tham số cũng ký hiệu `μ`, hãy đặt tên rõ như `mu_step`, `mu_seq` và chú thích. Tương tự, `rho` của bài toán tài chính không được dùng nhầm cho hệ số quán tính.

## 4. Hợp đồng hàm thuật toán (API)

Ưu tiên giao diện tối giản, độc lập dữ liệu:

```python
import numpy as np


def fbm(x0, B, prox, gamma, max_iter=10000,
        tol=1e-8, iter_list=(1, 10, 100, 1000, 10000)):
    """Giải 0 ∈ A(x) + B(x) bằng bước tiến–lùi.

    B(x): toán tử đơn trị.
    prox(y, gamma): J_(gamma*A)(y).
    """
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    res = {0: x.copy()}
    converged = False
    residual = float('inf')
    iterations = 0

    for k in range(1, max_iter + 1):
        x_new = np.asarray(
            prox(x - gamma * B(x), gamma), dtype=float
        ).reshape(-1)
        if x_new.shape != x.shape or not np.isfinite(x_new).all():
            raise ValueError('Nghiệm lặp không hợp lệ')
        residual = float(np.linalg.norm(x_new - x))
        x = x_new.copy()
        iterations = k
        if k in iter_list:
            res[k] = x.copy()
        if residual <= tol:
            converged = True
            break

    return {
        'x': x,
        'iterations': iterations,
        'converged': converged,
        'residual': residual,
        'res': res,
    }
```

**Lưu ý về tiêu chuẩn dừng:** `||x_new - x|| <= tol` ở ví dụ là **sai khác bước lặp**, không nhất thiết là chứng nhận nghiệm chính xác. Khi cần so sánh bài bản, nên dùng **phần dư điểm bất động có chuẩn hóa** (ví dụ `||x - prox(x - gamma*B(x), gamma)|| / gamma`) và/hoặc kiểm tra riêng `feasibility`. Tên và định nghĩa `residual` phải được ghi trong docstring; các thuật toán so sánh phải dùng cùng quy tắc được thống nhất trước.

**Giao diện trả về chuẩn** cho tất cả thuật toán: dictionary với tối thiểu các khóa `x`, `iterations`, `converged`, `residual`, `res`. Có thể thêm `time`, `objective_history`, `feasibility`, `stop_reason`, nhưng không được tự đổi tên khóa nền tảng. Tên hàm là `fbm`, `sa_ifb` v.v. (snake_case). Các thuật toán quán tính có thể cần `x1`, các chuỗi tham số hoặc trạng thái khác; ghi rõ bằng tham số của hàm, **không gán cứng bằng biến toàn cục**.

## 5. Ví dụ tối thiểu để kiểm thử thuật toán

```python
import numpy as np
from algorithms import fbm

F = np.array([[np.sqrt(3), 0], [0, 1], [2, 0],
              [0, np.sqrt(2)], [0, 2]], dtype=float)
z = np.array([3*np.sqrt(3), 1, 6, np.sqrt(2), 2], dtype=float)
x0 = np.array([100., -100.])


def proj_ball(y, center, radius):
    d = y - center
    norm_d = np.linalg.norm(d)
    if norm_d <= radius:
        return y.copy()
    return center + radius * d / norm_d


def B(x):
    return F.T @ (F @ x - z)


def prox(y, gamma):
    return proj_ball(y, np.zeros(2), 1.0)

L = np.linalg.norm(F, ord=2) ** 2
gamma = 1.0 / L
result = fbm(x0=x0, B=B, prox=prox, gamma=gamma)
x_star = result['x']
res = result['res']
```

- `proj_ball` là phép chiếu lên **quả cầu đóng**, không phải mặt cầu. Khi điểm đã nằm trong quả cầu, giữ nguyên. Không chia cho `0`.
- `gamma = 1/L` **chỉ là lựa chọn cho ví dụ** với gradient bình phương tối thiểu và hằng số Lipschitz `L` như trên. Với toán tử khác phải xác lập điều kiện cỡ bước riêng.
- Mỗi ví dụ toán học nên có nghiệm chuẩn (phân tích hoặc đối chứng bằng phương pháp độc lập) và kiểm tra sai số.

## 6. Phân tách với `VIP_full.ipynb`

**Trong notebook dữ liệu:** giữ nguyên cách nạp riêng từng bộ FF10/FF48/FF100, không ép gộp vào một loader. Các biến cấu hình giữ là `CLEAN` (`True`/`False`), `FILL` (`'nan'`/`'zero'`) và `PATH_FF10`, `PATH_FF48`, `PATH_FF100`. Dữ liệu từ Ken French có mã thiếu `-99.99` và `-999` (đơn vị phần trăm); xử lý missing **trước** khi chia lợi suất cho 100. `FILL='zero'` chỉ là giả định thử nghiệm, không phải lợi suất quan sát thực tế.

**Trong notebook tài chính:** chọn dữ liệu cố định hoặc từng Rolling Window, ước lượng `R`, `mu`, `Sigma`, `rho`, định nghĩa `B(x)` và `prox(y, gamma)` theo **đúng bài toán tối ưu đã chọn**, rồi gọi solver từ `algorithms.py`.

```python
from algorithms import fbm

# R, mu, Sigma, rho, x0 đã chuẩn bị từ dữ liệu được chọn.
# B và prox phải được định nghĩa theo mô hình toán học hiện hành.
result = fbm(x0=x0, B=B, prox=prox, gamma=gamma,
             max_iter=10000, tol=1e-8)
x_star = result['x']
```

**Chú ý toán học:** Nếu `C = {x >= 0, sum(x)=1, mu@x=rho}` thì `||x||_1 = 1` trên `C`; cộng `lam * ||x||_1` vào hàm mục tiêu **không tạo được tính thưa chỉ nhờ hạng tử đó**. Không được nhận định `lam` tự động làm nghiệm thưa nếu vẫn giữ các ràng buộc này. Với `A = ∂(lam||x||_1 + ι_C)` trong trường hợp này, resolvent tương ứng chính là phép chiếu `P_C`; với mô hình/ràng buộc khác, `prox` có thể khác.

## 7. Chuẩn kiểm tra và báo cáo kết quả

Một thuật toán được coi là tích hợp xong khi:

- Chạy độc lập trên ví dụ NumPy nhỏ, kiểm tra shape, không phát sinh `NaN/Inf`.
- Không làm thay đổi `x0`, `R`, `mu`, `Sigma` ở bên ngoài hàm.
- Lưu snapshot đúng ở các vòng lặp được yêu cầu (`res[k] = x.copy()`).
- In hoặc trả đầy đủ `x`, `iterations`, `converged`, `residual` và (nếu có) `feasibility`.
- Xác định rõ **công thức phần dư**, cỡ bước, điều kiện dừng và điều kiện bảo đảm hội tụ của bài toán.
- So sánh các thuật toán trên **cùng** `B`, `prox`, dữ liệu, điểm khởi tạo và tiêu chuẩn dừng khi có thể.
- Khi làm thực nghiệm ngoài mẫu, không dùng `R_test` để tính `mu`, `Sigma`, chọn `lam` hoặc khởi tạo dựa trên lợi suất tương lai.
- Khi có thiếu dữ liệu, không âm thầm bỏ dòng, điền `0` hoặc nội suy. Quy tắc xử lý phải do phần dữ liệu cấu hình và báo cáo rõ.

## 8. Quy tắc làm việc với agent khác

Khi yêu cầu một agent viết thêm thuật toán, hãy gửi nội dung sau:

> Hãy đọc và tuân thủ `README.md` của dự án ĐA2. Chỉ dùng NumPy và các hàm `def` đơn giản nếu khả thi. Viết solver độc lập dữ liệu tài chính, dùng vector 1 chiều `(N,)`, tham số `B`, `prox`, `gamma`, `x0`, `max_iter`, `tol`, `iter_list`; trả về dictionary có `x`, `iterations`, `converged`, `residual`, `res`. Không đổi tên biến/API tùy tiện. Giải thích công thức lặp và chọn cỡ bước đúng với thuật toán; bổ sung ví dụ NumPy nhỏ để kiểm tra. Khi thuật toán quán tính cần tham số bổ sung, khai báo chúng tường minh và chú thích ý nghĩa.

---

**Nguyên tắc chính:** *Một bài toán có thể được giải bằng nhiều thuật toán; một thuật toán có thể áp dụng cho nhiều bài toán.* Hãy giữ **lớp dữ liệu**, **lớp mô hình/toán tử** và **lớp solver** tách biệt.
