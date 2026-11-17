# Báo cáo QC tự động (nháp)

> **Gợi ý** do script tính; bạn xác nhận. Script chỉ đếm theo phiếu kiểm bằng mắt người.

## Mẫu
- 7 ảnh (2 ca khó + 5 ngẫu nhiên), 153 đối tượng.
- taxonomy v1.1

## Số đo
| Chỉ số | Giá trị | Mẫu số |
|---|---|---|
| Lỗi **theo đối tượng** | 5/153 = 3.3% | 153 đối tượng trong mẫu |
| **Ảnh có lỗi** (dùng cho ngưỡng) | 4/7 = 57.1% | 7 ảnh trong mẫu |
| Máy: accepted / edited / deleted | 491/557 = 88.2% / 9/557 = 1.6% / 55/557 = 9.9% | 557 box máy |
| Máy: added | 16/516 = 3.1% | 516 đối tượng sau duyệt |
| Bắt lỗi bẫy | 8/12 = 66.7% | 12 bẫy |

## Top 3 loại lỗi
| Loại lỗi | Số lỗi | Ảnh ví dụ |
|---|---|---|
| sai_vi_tri_kich_thuoc | 2 | img_006.jpg, img_009.jpg |
| thua | 2 | img_024.jpg, img_034.jpg |
| bo_sot | 1 | img_024.jpg |

## Giới hạn thống kê
- 4 ảnh lỗi trong 7 ảnh mẫu: cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là **84%**. Ngưỡng 2% không thể được **chứng minh** bằng mẫu nhỏ như vậy: hãy nói rõ giới hạn này trong báo cáo.

## Gợi ý quyết định
**REWORK**

- ảnh có lỗi 57.1% > 2%
- bắt bẫy 67% < 80% -> lô duyệt chưa đáng tin

Cây quyết định: Hold (guideline chưa rõ / dữ liệu hỏng) -> Rework (> ngưỡng, lỗi lặp, bắt bẫy / nhất quán thấp) -> Release.
