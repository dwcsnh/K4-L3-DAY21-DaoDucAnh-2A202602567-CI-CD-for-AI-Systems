# Báo Cáo Lab Day 21 - CI/CD cho AI Systems


| | |
|---|---|
| Họ và tên | Đào Đức Anh |
| MSSV | 2A202602567 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/dwcsnh/K4-L3-DAY21-DaoDucAnh-2A202602567-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ siêu tham số ở lần 3 đạt F1-score cao nhất (0.7149), vượt ngưỡng chất lượng 0.65 của bài toán. Đáng chú ý, lần chạy có accuracy cao nhất là lần 1 (0.8780) lại không phải là lần có F1 cao nhất (0.7109 so với 0.7149). Điều này chứng minh accuracy bị chi phối bởi lớp đa số (<= 50K chiếm ~75%), trong khi F1 phản ánh đúng năng lực phân loại của lớp thiểu số (> 50K). Việc tăng n_estimators lên 200 và max_depth lên 5 giúp Gradient Boosting học được các mối quan hệ phi tuyến sâu hơn giữa các thuộc tính kinh tế - xã hội, mang lại F1-score tối ưu nhất.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult có phân bố mất cân bằng rõ rệt khi lớp thu nhập dương (> 50K) chỉ chiếm 24.8%, còn lớp âm (<= 50K) chiếm tới 75.2%. Nếu một mô hình đơn giản chỉ dự đoán "thu nhập thấp" cho mọi mẫu dữ liệu, nó vẫn dễ dàng đạt được accuracy 75.2% dù hoàn toàn vô dụng và không hề học được tri thức nào từ dữ liệu. 

Chỉ số F1-score cho lớp dương là trung bình điều hòa giữa Precision và Recall riêng biệt của lớp thiểu số, đo lường chính xác khả năng mô hình phát hiện người có thu nhập cao mà không bị pha loãng bởi độ chính xác trên lớp đa số. Lab này tuyệt đối không dùng `average="weighted"` hay `average="macro"` vì các cách tính trung bình này sẽ cộng dồn trọng số của lớp đa số, làm lu mờ hoàn toàn mục tiêu cốt lõi là đánh giá khả năng nhận diện lớp thiểu số.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lỗi build `scikit-learn` và import SQLAlchemy khi cài dependencies | Môi trường ảo dùng Python 3.14 chưa có wheel và SQLAlchemy 2.1 loại bỏ class nội bộ của MLflow | Tạo lại venv với Python 3.11 và ghim `sqlalchemy<2.1` trong requirements.txt |
| Không thể kết nối SSH vào máy ảo EC2 (`Connection timed out`) | Route `0.0.0.0/0` trong VPC Route Table bị rơi vào trạng thái `blackhole` do Internet Gateway bị detached | Dùng AWS CLI gắn lại Internet Gateway `igw-0b4a2cb9ed778b169` vào VPC |
| Lỗi unpickle model trên EC2 khi gọi API inference | Phiên bản scikit-learn trên EC2 cài mặc định (1.7.2) không tương thích ngược với model train ở 1.4.2 | Cài đặt lại chính xác phiên bản `scikit-learn==1.4.2` trên máy ảo EC2 |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu dữ liệu mới từ batch 2 (tổng cộng 44.722 mẫu), F1-score tăng từ 0.7149 lên 0.7354 và accuracy tăng từ 0.8740 lên 0.8820. Do batch 2 cùng phân phối với batch 1, sự cải thiện cho thấy mô hình Gradient Boosting tận dụng thêm các mẫu để làm mượt ranh giới quyết định của lớp thiểu số. Điều then chốt nhất là pipeline CI/CD đã tự động kích hoạt hoàn hảo: huấn luyện lại, vượt qua Quality Gate, và reload mô hình mới lên server mà không cần bất kỳ thao tác thủ công nào.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [x] Bonus 1 - Tracking MLflow từ xa với DagsHub: Tích hợp cấu hình nhận diện `MLFLOW_TRACKING_URI/USERNAME/PASSWORD` từ secret vào `train.py` và `cicd.yml`.
- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: Quét ngưỡng [0.1, 0.9] bước 0.05, xác định ngưỡng tối ưu 0.30 giúp tăng F1 từ 0.7354 lên 0.7537 và log vào MLflow.
- [x] Bonus 3 - Báo cáo precision / recall tự động: Tạo `outputs/detail.txt` chứa confusion matrix và per-class metrics, lưu qua GitHub Actions artifact (với bài toán này bỏ sót người thu nhập cao tốn kém hơn vì mất khách hàng mục tiêu).
- [x] Bonus 4 - Hoàn trả về phiên bản trước: Đọc `report.json` của model cũ từ S3; nếu F1 mới < F1 cũ thì Quality Gate tự động fail dừng pipeline để bảo vệ model đang chạy.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: Tự động kiểm tra tỷ lệ lớp dương trong tập huấn luyện (đạt 0.2478, lệch 0.02% <= 5% so với mốc 0.2480) và cảnh báo drift nếu lệch > 5%.
