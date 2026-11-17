# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 556; đối tượng sau duyệt: 518; ngưỡng accepted: IoU >= 0.9
- Đã **loại 26 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 486 | 486/556 = 87.4% | 556 box máy |
| edited (sửa) | 15 | 15/556 = 2.7% | 556 box máy |
| deleted (xoá) | 55 | 55/556 = 9.9% | 556 box máy |
| added (vẽ thêm) | 17 | 17/518 = 3.3% | 518 đối tượng sau duyệt |

Trong số deleted + added, **8 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 9.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 285 | 2 | 8 | 4 |
| bicycle | 6 | 0 | 1 | 1 |
| motorbike | 141 | 3 | 9 | 4 |
| car | 41 | 3 | 9 | 1 |
| bus | 3 | 0 | 0 | 4 |
| truck | 1 | 0 | 5 | 0 |
| three_wheeler | 9 | 7 | 23 | 3 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 67 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 63 | 11 | 4 | 85.1% (KTC 95%: 75–91%) | 94.0% (KTC 95%: 86–98%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 11/12 = 91.7% (KTC 95%: 65–99%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - extra: 3/3
  - missing: 3/3
  - wrong_class: 3/3
  - loose: 2/3
  - lọt: img_023.jpg (loose)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
