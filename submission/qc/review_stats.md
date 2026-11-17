# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 557; đối tượng sau duyệt: 516; ngưỡng accepted: IoU >= 0.9
- Đã **loại 24 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 491 | 491/557 = 88.2% | 557 box máy |
| edited (sửa) | 9 | 9/557 = 1.6% | 557 box máy |
| deleted (xoá) | 55 | 55/557 = 9.9% | 557 box máy |
| added (vẽ thêm) | 16 | 16/516 = 3.1% | 516 đối tượng sau duyệt |

Trong số deleted + added, **6 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 10.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 280 | 0 | 10 | 3 |
| bicycle | 5 | 0 | 3 | 2 |
| motorbike | 145 | 0 | 6 | 2 |
| car | 49 | 0 | 6 | 1 |
| bus | 3 | 0 | 0 | 3 |
| truck | 2 | 0 | 4 | 1 |
| three_wheeler | 7 | 9 | 26 | 4 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 67 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 63 | 11 | 4 | 85.1% (KTC 95%: 75–91%) | 94.0% (KTC 95%: 86–98%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 8/12 = 66.7% (KTC 95%: 39–86%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - wrong_class: 3/3
  - loose: 0/3
  - missing: 2/3
  - extra: 3/3
  - lọt: img_022.jpg (loose), img_009.jpg (loose), img_035.jpg (missing), img_006.jpg (loose)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
