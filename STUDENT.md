# Thông tin học viên

- Họ và tên: Nguyen Van A (Thay bang ho ten cua ban)
- MSHV: 000000 (Thay bang MSHV cua ban va chay lai inject_traps.py neu doi)
- Lớp (L2 / L3 / khác): L2/L3 (Thay lop cua ban)
- Tài khoản GitHub: your-github-username (Thay tai khoan cua ban)
- Repo bài nộp (URL): https://github.com/your-github-username/K5-L23-DAY08-000000-NguyenVanA-AIPrelabel
- Commit chốt bài (hash): xem git log HEAD sau commit cuoi (cap nhat truoc khi nop)
- Máy: Windows (Python 3.12.16 venv, GPU NVIDIA GeForce RTX 4050 Laptop GPU 6GB CUDA)
- CVAT (Docker): Chua cai/chua dung trong phien AI lam ho (mo phong offline qua COCO; cap nhat neu ban chay CVAT)

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: `491/557 (88.2%) / 9/557 (1.6%) / 55/557 (9.9%) (mẫu số: 557 box máy)`; added: `16/516 (3.1%) (mẫu số: 516 đối tượng sau duyệt; trong đó 6 ca đổi lớp -> sót thật ~10)`
- So với gold: precision / recall của **máy** `85.1% (63/74) / 94.0% (63/67)` *(lạc quan do gold dùng pre-label làm nền và chọn ngưỡng in-sample)*
- Bắt bẫy: `8/12 = 66.7%` (lọt 3/3 `loose`, 1/3 `missing`)
- Thời gian: gán tay `0.75 phút/ảnh (3 phút / 4 ảnh, AI làm bằng pre-label làm nền)`; duyệt có pre-label `0.25 phút/ảnh (6 phút / 24 ảnh, AI duyệt qua overlay)`
- Lỗi QC: `3.3% theo đối tượng (5/153)`, `57.1% theo ảnh (4/7)`
- **Quyết định:** `Rework` — lý do một dòng: `Ảnh lỗi mẫu QC 4/7 = 57.1% (> 2%) và bắt bẫy 8/12 = 66.7% (< 80%, lọt 3/3 loose), cần duyệt lại đủ bước 4 checklist trước khi Release.`
