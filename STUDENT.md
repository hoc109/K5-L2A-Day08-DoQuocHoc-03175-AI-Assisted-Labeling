# Thông tin học viên

- Họ và tên: Do Quoc Hoc
- MSHV: 2A202603175
- Lớp (L2 / L3 / khác): L2A
- Tài khoản GitHub: hoc109
- Repo bài nộp (URL): https://github.com/hoc109/K5-L2A-Day08-DoQuocHoc-03175-AI-Assisted-Labeling
- Commit chốt bài (hash): cập nhật sau commit cuối (xem `git rev-parse --short HEAD`)
- Máy: Windows (Python 3.12.16 venv, GPU NVIDIA GeForce RTX 4050 Laptop GPU 6GB CUDA)
- CVAT (Docker): CVAT 2.74.1 bản chuẩn Docker chạy local tại http://localhost:8080 (không cài Nuclio/SAM serverless; đã tạo Project `day08-2A202603175` `id=3` gồm 3 Task: `gold-manual` #10 [4 ảnh, 68 shapes], `batch-review` #11 [24 ảnh, 532 shapes], `batch-prelabel` #12 [24 ảnh, 570 shapes])

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: `486/556 (87.4%) / 15/556 (2.7%) / 55/556 (9.9%) (mẫu số: 556 box máy)`; added: `17/518 (3.3%) (mẫu số: 518 đối tượng sau duyệt; trong đó 8 ca đổi lớp -> sót thật ~9)`
- So với gold: precision / recall của **máy** `85.1% (63/74) / 94.0% (63/67)` *(lạc quan do gold dùng pre-label làm nền và chọn ngưỡng in-sample)*
- Bắt bẫy: `11/12 = 91.7%` (Vòng 2 Rework với MSHV `2A202603175`: `extra 3/3`, `missing 3/3`, `wrong_class 3/3`, `loose 2/3`; lọt 1 bẫy `img_023.jpg loose`; tăng từ `8/12 = 66.7%` ở Vòng 1)
- Thời gian: gán tay `0.75 phút/ảnh (3 phút / 4 ảnh, AI dùng pre-label làm nền)`; duyệt có pre-label `0.58 phút/ảnh (14 phút / 24 ảnh gồm 6 phút Vòng 1 + 8 phút Vòng 2 Rework phóng to/lưới 100px)`
- Lỗi QC: `0.0% theo đối tượng (0/182)`, `0.0% theo ảnh (0/8)` (Vòng 2 sau Rework; Vòng 1 trước khi sửa là `5/153 = 3.3%` đối tượng và `4/7 = 57.1%` ảnh)
- **Quyết định:** `Release` — lý do một dòng: `Sau vòng Rework cho MSHV 2A202603175, mẫu QC đạt 0/8 ảnh lỗi (0.0% <= 2%), bắt bẫy đạt 11/12 = 91.7% (>= 80%), guideline đã chốt v1.1 (lưu ý giới hạn mẫu nhỏ: cận trên Wilson 95% = 32%).`
