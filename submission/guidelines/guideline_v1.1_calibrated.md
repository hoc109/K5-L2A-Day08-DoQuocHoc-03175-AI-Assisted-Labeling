# Label contract — Phố châu Á — BỔ SUNG v1.1 — 000000 - Nguyen Van A (Thay MSHV va Ten cua ban)

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Các quyết định dưới đây khớp với `submission/qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Nếu không có bàn đạp, yên liền thân xe dạng xe tay ga điện hoặc xe thồ có động cơ → `motorbike`. Chỉ gán `bicycle` khi thấy bàn đạp cơ học rõ ràng. Ca không nhìn rõ bàn đạp/động cơ → giữ lớp tin nhất và tick `can_review`. | `img_009.jpg`, `img_025.jpg` | `CAL-01` |
| 2 | Khối xe máy đỗ dày đặc gần/xa camera: từng xe hay `ignore_region`? | Các xe ở hàng ngoài cùng tách biệt rõ (nhìn thấy ≥ 50% thân/bánh, chiều cao ≥ 20 px) → gán từng `motorbike`. Các hàng phía sau đỗ chồng lớp dày đặc (≥ 5 xe không tách được ranh giới từng chiếc hoặc chiều cao < 20 px) → gom thành một `ignore_region`. | `img_026.jpg`, `img_018.jpg` | `CAL-02` |
| 3 | Hàng hoá chở trên xe (tre, rổ, hoa quả): box và mask có gồm hàng? | Hàng đặt gọn trong giỏ/thùng gắn cố định vào xe (chuối trong thùng xe ba gác, thúng hàng trên giá xe đạp) **tính vào** cả box và mask xe. Chỉ phần hàng cồng kềnh nhô ra khỏi thân/thùng xe > 20% chiều dài xe mới cắt ra ngoài box. | `img_008.jpg`, `img_031.jpg` | `CAL-03` |
| 4 | Người ngồi trong tuk-tuk có mui che: ngưỡng ≥ 20% diện tích có hợp lý? | Giữ ngưỡng 20% ở mục 4.1.2, nhưng cụ thể hoá để giảm lệch giữa người gán: chỉ gán `person` khi nhìn thấy rõ **ít nhất phần đầu+vai hoặc nửa thân** tách biệt (không gán bóng mờ qua bạt). Ca mấp mé ngưỡng tick `can_review`. | `img_012.jpg`, `img_013.jpg` | — |
| 5 | Kính chắn gió / cửa sổ xe có thuộc mask xe? | Thuộc mask `car` / `bus` / `three_wheeler` nếu là tấm kính gắn trên khung xe (trừ khoảng hở hoàn toàn không kính ở hai bên thành tuk-tuk hay xích lô, nơi nhìn xuyên thấy cảnh nền/người phía sau). | `img_013.jpg` | — |
| 6 | Box máy chồng nhau nhiều trong ảnh đêm / ngã tư đông: gộp hay giữ? | Tách từng đối tượng rõ nét ở tiền cảnh; các vệt đèn nhoè phơi sáng dài không nhận ra hình dạng vật thật thì **xoá** (`deleted`); cụm xe ở xa bị nhoè đèn chồng chéo ≥ 5 vật → gộp `ignore_region`. | `img_024.jpg` | `CAL-02` |
| 7 | Có đặt mức `conf` tối thiểu để chấp nhận không cần sửa? | **Không** đặt ngưỡng `conf` miễn kiểm tra. Sau khi lọt 4/12 bẫy (gồm 3/3 `loose`) với tỷ lệ `accepted` 88.2%, `conf` chỉ dùng để tham khảo thứ tự ưu tiên, mọi box vẫn phải qua đủ 4 bước checklist. | `img_006.jpg`, `img_022.jpg` | `CAL-05` |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_001.jpg`, `img_012.jpg` | Xe bán tải 4 bánh và xe tuk-tuk bị nhầm với `truck` / `car` | YOLO26m gán xe bán tải thành `truck` (`0.88` ở `img_001`), gán tuk-tuk thành `car` (`img_012`, `img_013`) | Xe bán tải thân liền gia dụng gán `car`; xe ba bánh có mui/thùng hàng gán `three_wheeler`; chỉ xe tải thùng hàng ≥ 4 bánh mới gán `truck` | Mục 2 (`car`, `truck`, `three_wheeler`) và Mục 5 (`CAL-04`) |
| `img_002.jpg`, `img_014.jpg` | Gánh hàng rong, quầy sạp bán hoa quả bị YOLOE ở ngưỡng `0.15` bắt nhầm | YOLOE-26s gán `three_wheeler` cho quang gánh (`img_014`) và bàn sạp/rào chắn (`img_012`, `img_022`) | Xoá toàn bộ box không có bánh xe tự hành (quang gánh, quầy sạp cố định, thùng hàng rời) | Mục 2 (`three_wheeler`, `CAL-04`) |
| `img_024.jpg` | Xe khách 16 chỗ (minibus) ban đêm bị nhận nhầm giữa `bus` và `truck` | YOLO26m gán chiếc bên trái là `bus` (`0.87`), chiếc tương tự bên phải là `truck` (`0.53`) | Thống nhất gán `bus` cho mọi xe chở khách từ 12–16 chỗ trở lên (minibus/van chở khách) | Mục 2 (`bus`, `CAL-06`) |
| `img_006.jpg`, `img_009.jpg`, `img_022.jpg`, `img_035.jpg` | Bỏ sót 3/3 bẫy `loose` và 1/3 bẫy `missing` do duyệt trên ảnh chưa phóng to | Pre-label + bẫy trộn lẫn; người duyệt (AI) chấp nhận nhầm 3 box hơi lệch/lỏng | Bắt buộc kiểm tra độ ôm sát biên (≤ ~2 px ở ảnh 1024 px) khi phóng to ở bước 4 checklist; trả lô về **Rework** | Mục 4.1.1 & Mục 7 (`CAL-05`) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt 7 câu hỏi mở (xe đạp điện, bãi xe `ignore_region`, hàng hoá trên xe, ngưỡng che 20%, kính xe, vệt mờ ban đêm, không miễn kiểm theo `conf`); bổ sung phân biệt bán tải/minibus/quầy hàng và quy tắc kiểm biên chống `loose` | `CAL-01`, `CAL-02`, `CAL-03`, `CAL-04`, `CAL-05`, `CAL-06` |
