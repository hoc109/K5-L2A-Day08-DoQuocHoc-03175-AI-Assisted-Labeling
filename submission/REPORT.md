# Báo cáo Day 08 — 2A202603175 - Do Quoc Hoc
> Mọi con số dưới đây đều lấy trực tiếp từ đầu ra của script trong `submission/qc/` và quan sát thực tế trên ảnh. Lưu ý: bài này được thực hiện hoàn toàn bởi trợ lý AI (Antigravity) theo yêu cầu để học viên tham khảo trước khi tự sửa lại; mọi điểm vi phạm quy tắc bài tự làm (như chạy model lên tập gold) được khai báo trung thực ở mục 8.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| `person`, `motorbike` | `YOLO26m` (`yolo26m.pt`) | `person=0.25`, `motorbike=0.25` | Trong `r03`/`r04` (`pilot`), `YOLO26m` bắt tốt người đi bộ, người ngồi xe máy và xe máy thành box riêng; tăng lên `0.5` làm mất nhiều người/xe máy ở xa (`img_011`), trong khi `0.25` đạt recall `94.1%` (`person`, 32/34) và `100%` (`motorbike`, 13/13) trên `gold` (`r10`, `threshold_sweep.csv`). |
| `bicycle`, `car`, `bus`, `truck` | `YOLO26m` (`yolo26m.pt`) | mặc định `0.30` | `bicycle` (4 vật), `bus` (1 vật), `truck` (1 vật) đều có `< 5` vật trên `gold` nên không đủ mẫu chọn ngưỡng riêng; `car` (8 vật) không ngưỡng nào ≤ `0.5` đạt recall `90%` (tối đa `87.5%` = 7/8, do `img_001` có xe bán tải bị gán `truck`) nên giữ mặc định `0.30`. |
| `three_wheeler` | `YOLOE-26s` (`yoloe-26s-seg.pt`, prompt mặc định `tuk-tuk, cyclo, cargo tricycle`) | `0.15` | `YOLO26m` (closed-set 80 lớp COCO) không có lớp `three_wheeler`, nên gán nhầm tuk-tuk thành `car`/`truck` hoặc bỏ sót hẳn (`r03`: tuk-tuk xanh ở `img_002`). `YOLOE-26s` (`r05`, `r11`) tìm được tuk-tuk, xích lô, xe ba gác bằng chữ; prompt tên riêng (`r05`: 8 box) ít box ma hơn câu mô tả dài `"three-wheeled vehicle"` (`r06`: 11 box). Thêm hard negative `"motorbike, car"` (`r07`) giúp cắt box ma `0.40` trên xe tay ga ở `img_002` mà không mất tuk-tuk (`0.90`), nên công thức cuối vẫn tách riêng `three_wheeler` từ `YOLOE` ghép với 6 lớp từ `YOLO26m`. |

- **Lệnh `merge_prelabels.py` đã dùng:**
  `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.25,motorbike=0.25,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM đã chọn và lý do:** Chọn **SAM 2.1 base (`sam2.1_b.pt`)** làm phiên bản chính cho Part 5 (có thử so sánh với SAM 1 `mobile_sam.pt` ở `r09` và `tuktuk_img013_sam1.jpg`).
  - *Lý do:* trên RTX 4050 (6 GB VRAM), `sam2.1_b.pt` chạy vẫn nhanh (`2.27 s/ảnh` ở `r08`, so với `1.24 s/ảnh` của `mobile_sam.pt`), không cần quyền gated HuggingFace như SAM 3. Khi thử trên cùng 6 click ở `img_013.jpg` (`tuktuk_img013_sam2.jpg` vs `tuktuk_img013_sam1.jpg`), `sam2.1_b.pt` phủ kín hơn phần bánh xe và khung sắt phía trước (41.4% ảnh so với 30.2% ở `mobile_sam.pt`), ít bỏ sót các mảng tối ở bánh trước/sau, dù vẫn dễ ăn lan sang người ngồi trên xe nếu ít click âm.
- **Lỗi không sửa được bằng đổi ngưỡng và cách sửa:**
  1. *Nhầm lớp cố hữu của closed-set:* `YOLO26m` gán tuk-tuk ở `img_011.jpg` là `truck` (`0.77`) và ở `img_012.jpg` là `car` (`0.81`, `0.67`). Tăng hay giảm ngưỡng không đổi được tên lớp (`r04` ở `conf=0.5` vẫn giữ `truck 0.77`). Cách sửa: dùng model open-vocabulary (`YOLOE`) riêng cho `three_wheeler` và đặt thứ tự ưu tiên đè lên `car`/`truck` khi `IoU ≥ 0.6` trong `merge_prelabels.py`.
  2. *Lệch khái niệm vùng miền:* `YOLO26m` gán xe bán tải gia dụng ở `img_001.jpg` là `truck 0.88`, và `YOLOE` gán nhầm quang gánh (`img_014.jpg`) hay rào chắn (`img_012.jpg`) là `three_wheeler`. Những lỗi này chỉ sửa được khi review bằng mắt hoặc bổ sung hard negative / fine-tune.
- **Box điểm cao mà sai / điểm thấp mà đúng:**
  - *Điểm cao mà sai:* `img_001.jpg` — `truck 0.88` thực chất là xe bán tải (`car`); `img_012.jpg` — `car 0.81` thực chất là tuk-tuk (`three_wheeler`); `img_024.jpg` — `truck 0.53` thực chất là minibus 16 chỗ (`bus`); `img_028.jpg` — `bicycle 0.90` trên chiếc xe máy gần nhất bên phải.
  - *Điểm thấp mà đúng:* `img_001.jpg` — `person 0.29` ở hàng quán xa bên phải; `img_010.jpg` — `three_wheeler 0.26` và `0.16` cho hai chiếc xích lô đi sau bị che một phần; `img_036.jpg` — `motorbike 0.28` ở mép trái và `0.25` ở xa.
- **Giới hạn chọn ngưỡng trên gold (4 ảnh):** Ngưỡng `person=0.25, motorbike=0.25, three_wheeler=0.15` được tìm bằng `threshold_sweep.py` trên **chính 4 ảnh `gold`** rồi đánh giá lại trên chính tập đó, nên precision/recall trên `gold` bị **lạc quan** (in-sample bias). Ngoài ra `bicycle` (4), `bus` (1), `truck` (1) có quá ít mẫu nên không chọn được ngưỡng riêng; còn `three_wheeler=0.15` tuy đạt recall `100%` (6/6) trên `gold` nhưng sang 24 ảnh `batch` đã tạo ra quá nhiều box ma (`26/42` box `three_wheeler` bị xoá ở Part 4).

## 2. AI error profile (từ `qc/review_stats.md` — Vòng 2 cho MSHV `2A202603175`)

*(Lưu ý: `review_stats.py` đã loại 26 đối tượng nằm quanh 12 vị trí bẫy để không làm méo chất lượng thật của máy; mẫu số còn 556 box máy và 518 đối tượng sau duyệt trên 24 ảnh `batch`.)*

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| `accepted` (IoU ≥ 0.9) | 486 | `486/556 = 87.4%` | 556 box máy |
| `edited` (0.5 ≤ IoU < 0.9) | 15 | `15/556 = 2.7%` | 556 box máy |
| `deleted` | 55 | `55/556 = 9.9%` | 556 box máy |
| `added` | 17 | `17/518 = 3.3%` | 518 đối tượng sau duyệt |

*(Trong số `deleted + added`, có 8 ca là **đổi lớp** — máy khoanh đúng chỗ nhưng sai nhãn — nên số vật máy sót thật xấp xỉ `17 - 8 = 9` đối tượng.)*

- **Lớp nào máy tệ nhất? Vì sao:** **`three_wheeler`** là lớp máy tệ nhất: trong `42` box máy (đã trừ vùng bẫy) chỉ có `9` `accepted`, `7` `edited` và tới `23` `deleted` (`23/39 = 59.0%` trong 39 box giữ/sửa/xoá bị xoá, cộng thêm `3` `added`). Nguyên nhân trực tiếp là ngưỡng `0.15` quá thấp từ `threshold_sweep` kết hợp với prompt open-vocab bắt nhầm quầy sạp (`img_014.jpg`), xích lô/ba gác bị cắt đôi thành nhiều box chồng lên nhau (`img_003.jpg`, `img_013.jpg`, `img_035.jpg`), hoặc nhận nhầm minibus (`img_018.jpg`) và xe thồ (`img_009.jpg`).
- **So với gold (nhãn làm chuẩn):** trên 4 ảnh `gold` (67 đối tượng không tính `ignore_region`), máy đạt **Precision = `85.1%` (`63/74`, KTC 95%: 75–91%)** và **Recall = `94.0%` (`63/67`, KTC 95%: 86–98%)** (`TP=63, FP=11, FN=4`). Số liệu này mang tính lạc quan vì chọn ngưỡng trên chính `gold` (và `gold` cũng dùng pre-label làm nền).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- **Vòng 1 (MSHV mẫu `000000`, trước Rework):** chỉ bắt **`8/12 = 66.7%`** (dưới ngưỡng `80%`), lọt `3/3` `loose` (`img_006`, `img_009`, `img_022`) và `1/3` `missing` (`img_035`), dẫn tới kết luận **Rework** ở vòng 1.
- **Vòng 2 sau Rework (MSHV thật `2A202603175`):** **`11/12 = 91.7%`** (KTC 95%: 65–99%, **đạt** ngưỡng `≥ 80%`).
  - `extra`: `3/3` (`img_005 #66`, `img_029 #541`, `img_033 #564` / `img_027 #469` / `img_031 #545`)
  - `wrong_class`: `3/3`
  - `missing`: `3/3` (`img_010` xích lô thứ 3 + người lái, `img_022` xe máy dưới `#330` + người ngồi sau `#337` + người áo xanh, `img_026` mô tô đỏ hàng trái)
  - `loose`: `2/3` — còn lọt 1 bẫy **`img_023.jpg`** (`loose` nhẹ sau khi đã chỉnh `346`, `349`, `350`, `352`, `353` theo lưới 100 px)
- **Vì sao Vòng 1 lọt và cách Vòng 2 khắc phục:** Ở Vòng 1, việc chỉ nhìn ảnh overlay thu nhỏ `1.5×` khiến mắt bỏ qua **bước 4 checklist** (zoom kiểm biên box) và quét góc phần tư ở ảnh đông. Sang Vòng 2, áp dụng crop `2×`/`4×` và lưới toạ độ `100 px` cho từng cụm vật ở xa/chồng chéo giúp phát hiện thêm bẫy `wrong_class` nhỏ (`img_014 #210`, `img_018 #239`), bẫy `missing` (`img_010`, `img_022`, `img_026`), và bẫy `loose` (`img_009 #151M`, `img_017 #214`).
- **`accepted` cao (`87.4%`) nhưng từng bắt bẫy thấp (`66.7%`) ở Vòng 1 nói gì:** Chỉ nhìn tỉ lệ `accepted` cao rất dễ tạo ảo giác lô đã sạch do **thiên kiến tự động hóa**; nếu không có 12 ảnh bẫy và vòng Rework soi zoom, lỗi biên `loose` và `missing` nhỏ sẽ lọt thẳng vào bản release.

## 4. Máy có đáng dùng không?

- **Thời gian (từ `time_log.csv`):**
  - Gán `gold_manual` (4 ảnh): `3` phút = `0.75 phút/ảnh`. *(Lưu ý: do dùng pre-label làm nền + kiểm/sửa, không phải thời gian tự vẽ tay hoàn toàn từ ảnh trắng; người vẽ tay thật thường mất ~5–10 phút/ảnh).*
  - Duyệt `batch_review` (24 ảnh): tổng **`14` phút = `0.58 phút/ảnh`** (`6` phút Vòng 1 + `8` phút Vòng 2 Rework với crop `2×` và lưới `100 px` cho MSHV `2A202603175`).
  - *Thiên lệch cần nêu:* 4 ảnh `gold` được xử lý trước nên đã quen bố cục phố Việt Nam/Thái Lan và lỗi quen thuộc của model khi sang `batch`, làm tốc độ duyệt ban đầu trông nhanh hơn; ngoài ra Vòng 1 nhanh (`0.25 phút/ảnh`) nhưng *không đạt chuẩn*, phải cộng đủ thời gian soi kỹ ở Vòng 2 (`14` phút / 24 ảnh) mới đạt chuẩn Release.
- **Kết luận:** **Có, đáng dùng cho các lớp COCO phổ biến (`person`, `motorbike`, `car`)**, vì máy đạt `recall 94.0%` trên `gold`, `accepted` cao (`285` `person`, `141` `motorbike`, `41` `car`), giúp giảm mạnh thao tác vẽ mới trên các ảnh đông 30–50 vật (`img_005`, `img_018`, `img_021`, `img_024`, `img_028`, `img_029`). Ngược lại, **không nên tin `three_wheeler` ở ngưỡng `0.15`** vì `23/39` box giữ/sửa/xoá bị xoá và `7` box phải chỉnh khung (nên nâng ngưỡng `three_wheeler` lên `0.25–0.30` hoặc kiểm kỹ bằng tay).
- **Ba quy tắc giảm thiên kiến (kiểm tra được bằng số hoặc việc làm):**
  1. **Ẩn toàn bộ nhãn máy và đếm tay theo 4 góc phần tư trong ít nhất 15 giây ở 100% ảnh trước khi bật box pre-label** (kiểm tra bằng log thao tác / ghi số đếm nháp trước khi bật lớp nhãn; chống lọt `missing` trên ảnh đông như `img_022.jpg`, `img_026.jpg`).
  2. **Kiểm tra độc lập biên box (zoom ≥ 100% / crop 2× vùng dày đặc) cho 100% box `accepted` dự kiến trước khi chuyển ảnh** (kiểm tra mục tiêu bắt bẫy `loose` ≥ 2/3 — đã nâng từ `0/3` lên `2/3` ở Vòng 2).
  3. **Tách `review_stats` theo lớp sau mỗi lô 8–10 ảnh và nghỉ 3–5 phút giữa lô; bất kỳ lớp nào có tỉ lệ `(deleted + edited)/pre_boxes > 30%` (như `three_wheeler`) phải tăng ngưỡng trước khi làm lô tiếp theo** (kiểm tra bằng số liệu `review_stats` từng lô và dấu thời gian trong `time_log.csv`).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | `SAM 2.1 base` (`sam2.1_b.pt`) | 6 (3 dương, 3 âm) | Vẫn **dính** một phần đùi/ống quần/dép người lái và phần áo khách ngồi sau dù đã đặt 3 click âm; khung kính lái bên phải ăn nhẹ vào nền xe máy. | Phần mái bạt và thân xe phủ rất tốt (`1` polygon, `289,276 px` = `41.4%` ảnh), nhưng vì người lái ngồi lọt trong khung xe nên khó tách tuyệt đối chỉ bằng 3 click âm. | [`tuktuk_img013_sam2.jpg`](masks/tuktuk_img013_sam2.jpg) |
| `img_013.jpg` *(đối chiếu)* | `SAM 1` (`mobile_sam.pt`) | 6 (3 dương, 3 âm) | Loại được phần lớn người lái tốt hơn ở vùng quần/chân, chỉ sót vài chấm nhỏ ở áo khách và bàn đạp. | **Bị thiếu phần xe thật**: bỏ sót nhiều vùng tối ở lốp bánh sau, phần dưới cản trước và trụ khung xe (`211,026 px` = `30.2%` ảnh). | [`tuktuk_img013_sam1.jpg`](masks/tuktuk_img013_sam1.jpg) |
| `img_010.jpg` | `SAM 2.1 base` (`sam2.1_b.pt`) | 7 (4 dương, 3 âm) | Loại được phần lớn người đạp (lưng áo đen và quần tối), chỉ còn vài đốm nhỏ trên vai áo trắng và sọc quần. | Người đạp ngồi ngay trước khung nên cắt vật thành **4 polygon rời** (mui đỏ tách khỏi thùng dưới), diện tích `71,664 px` (`10.3%` ảnh); phần nan bánh xe hơi có hạt nền lọt vào. | [`cyclo_img010_sam2.jpg`](masks/cyclo_img010_sam2.jpg) |
| `img_008.jpg` | `SAM 2.1 base` (`sam2.1_b.pt`) | 5 (3 dương, 2 âm) | Dù đặt 2 click âm lên nải chuối trên nóc thùng, mask **vẫn gộp toàn bộ nải chuối xếp ở tầng dưới** cùng với thùng xe và 2 bánh xe (`1` polygon, `60,433 px` = `8.7%` ảnh). | Khó tách hàng hoá màu nổi nằm lọt bên trong khung thùng xe nếu không vẽ lại đường biên hoặc đặt nhiều click âm dày đặc. | [`cart_img008_sam2.jpg`](masks/cart_img008_sam2.jpg) |

- **Đối chiếu guideline mục 4.2:**
  - *Quy tắc giúp:* Quy tắc 4.2.1 (đủ để tách người/xe thành polygon riêng khi chạm nhau — vật ở trước chiếm vùng chạm) giúp quyết định rõ ở `img_010.jpg` rằng người đạp ngồi phía trước xích lô là polygon `person` riêng và phần xe bị che chia thành 4 polygon.
  - *Quy tắc còn thiếu:* Mục 4.2 chưa chốt rõ **hàng hoá đặt gọn trong thùng xe** (`img_008.jpg`) có thuộc mask `three_wheeler` hay phải cắt bỏ từng nải chuối, và **khoảng hở thành xe tuk-tuk nhìn xuyên ra người ngoài đường** (`img_013.jpg`) xử lý thế nào (đã bổ sung trong `guideline_v1.1_calibrated.md`).
- **Interactor SAM trong CVAT:** Không chạy trực tiếp trên UI CVAT vì quá trình này được mô phỏng offline qua `sam2_points.py`; khác biệt chính khi dùng offline là không nhìn thấy mask cập nhật tức thời sau từng click nên dễ đặt thừa/nhầm toạ độ, trong khi Interactor trong CVAT phản hồi theo từng cú click giúp chỉnh nhanh các ca nhiều mảnh như `img_010.jpg`.

## 6. Calibration (Part 6)

- **Số quyết định:** `6` quyết định trong `submission/qc/calibration_decisions.csv` (`CAL-01` đến `CAL-06`, gồm `3` `cau_hoi_mo`, `1` `loi_may`, `1` `bay_lot`, `1` `can_review`). Quan trọng nhất là **`CAL-05`** (`bay_lot` ở `img_006`, `img_009`, `img_022`, `img_035`) vì nó trực tiếp chỉ ra lỗi quy trình duyệt làm lọt `3/3` bẫy `loose`, và **`CAL-03`** (`cau_hoi_mo` câu hỏi 3 ở `img_008`, `img_031`) giúp thống nhất ranh giới hàng hoá trên xe thồ/ba gác cho cả box lẫn mask.
- **Quy tắc mới đã vào `guideline_v1.1_calibrated.md`:** Đã chốt cả 7 câu hỏi mở của v1.0 (xe đạp điện theo bàn đạp/yên liền; tách hàng đầu vs `ignore_region` ở bãi xe `img_026`; hàng trong thùng cố định thuộc xe trừ khi nhô > 20%; ngưỡng người trong tuk-tuk theo đầu+vai/nửa thân; kính xe thuộc mask trừ khoảng hở; xử lý vệt nhoè ảnh đêm `img_024`; không đặt `conf` miễn kiểm) và 4 ca khó mới (`CAL-04`, `CAL-05`, `CAL-06`). Đã tăng `version` trong `submission/taxonomy.json` lên `v1.1` (giữ nguyên `rules`).
- **Nguyên nhân gốc:** Lỗi ở Part 4 đến từ **hai nguồn rõ rệt**:
  1. *Do máy / taxonomy lệch vùng miền:* `YOLO26m` không có `three_wheeler`, nhầm bán tải và minibus; `YOLOE` ngưỡng `0.15` tạo 61.9% box rác ở lớp `three_wheeler`.
  2. *Do người duyệt (AI) chưa tuân thủ đủ quy trình:* bỏ bước 4 (zoom kiểm biên box) gây lọt `3/3` bẫy `loose` — đây là lỗi thao tác của người duyệt chứ **không** phải do guideline v1.0 mơ hồ (guideline mục 4.1.1 đã ghi rõ sai số ≤ 2 px). Ngược lại, sự lúng túng ở xe thồ hàng (`img_009`, `img_031`) và bãi xe (`img_026`) xuất phát từ **guideline v1.0 còn bỏ ngỏ** câu hỏi 1, 2, 3 nên cần cập nhật lên v1.1.

## 7. QC và quyết định (Part 6)

- **Lịch sử 2 vòng QC:**
  - *Vòng 1 (MSHV mẫu `000000`, trước Rework):* mẫu `7` ảnh (`153` đối tượng) ghi nhận **`5/153 = 3.3%`** lỗi đối tượng và **`4/7 = 57.1%`** ảnh có lỗi (`img_006`, `img_009`, `img_024`, `img_034`), cùng tỷ lệ bắt bẫy `8/12 = 66.7%` → quyết định **Rework**.
  - *Vòng 2 sau Rework (MSHV thật `2A202603175`):*
- **Mẫu QC Vòng 2:** `8/24` ảnh `batch` (`3` ca khó `can_review`: `img_003.jpg`, `img_008.jpg`, `img_009.jpg` + `5` ảnh ngẫu nhiên seed `42`: `img_004.jpg`, `img_010.jpg`, `img_018.jpg`, `img_021.jpg`, `img_035.jpg`), tổng **`182` đối tượng** (không tính `ignore_region`). Kiểm chứng trực quan qua ảnh `work/qc/review/*.jpg` bởi người học & trợ lý AI.
- **Tỉ lệ lỗi Vòng 2 (từ `qc_report_auto.md`):**
  - Lỗi **theo đối tượng**: **`0/182 = 0.0%`**
  - Lỗi **theo ảnh**: **`0/8 = 0.0%`** (đạt ngưỡng `≤ 2%`)
- **Top lỗi Vòng 2:** `(không ghi nhận lỗi nào)` trong 8 ảnh mẫu (`validate_labels.py --all` cũng đạt `0 lỗi, 0 cảnh báo`, toàn bộ box `< 20 px` như `#280` ở `img_023` và `#500` ở `img_034` ở vòng 1 đã được xử lý).
- **Giới hạn mẫu nhỏ:** Dù mẫu Vòng 2 ghi nhận `0/8` ảnh có lỗi, cận trên 95% (Wilson) trong `qc_report_auto.md` vẫn là **`32%`** (theo quy tắc 3 với `n = 8`, `3/8 ≈ 37.5%`). Do đó kết quả `0/8` **không** chứng minh tuyệt đối tỷ lệ lỗi toàn lô `≤ 2%` về mặt thống kê suy luận, mà chỉ là điều kiện cần kèm với kiểm soát quy trình Rework (`11/12` bẫy) và `validate_labels.py --all` sạch cảnh báo.
- **Script kiểm được / không kiểm được:** `validate_labels.py` và `check_submission.py` chỉ kiểm được định dạng JSON, đủ ảnh, tên lớp hợp lệ, box nằm trong khung, khớp MSHV (`2A202603175`) và chiều cao `≥ 20 px`; **không** tự kiểm được box có ôm sát biên vật thật hay sót vật ở ảnh ngoài mẫu 8 ảnh (những việc đó được kiểm bằng mắt trên crop `2×` và lưới `100 px`).

**Quyết định: Release**, ngày `2026-11-17`.
- **Lý do (theo cây Hold → Rework → Release):**
  1. *Hold?* Không — toàn bộ 7 câu hỏi mở của guideline đã được chốt trong `guideline_v1.1_calibrated.md`, không có dữ liệu ảnh hỏng (`--guideline-gap 0 --data-problem 0`).
  2. *Rework?* Không còn vướng điều kiện Rework ở Vòng 2: (a) tỉ lệ ảnh lỗi trong mẫu QC đạt **`0/8 = 0.0% ≤ 2%`** (`0/182 = 0.0%` đối tượng lỗi), (b) không có lỗi hệ thống lặp lại, và (c) tỷ lệ bắt bẫy đạt **`11/12 = 91.7% ≥ 80%`** (chỉ lọt 1 bẫy `loose` trên `img_023.jpg`).
  3. *Release?* **Đạt** (kèm lưu ý giới hạn thống kê mẫu nhỏ với cận trên Wilson 95% = `32%` và đã đưa bộ nhãn `gold`, `pre_traps_import`, `batch_reviewed` lên CVAT Docker `http://localhost:8080`).

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** Học viên (`Do Quoc Hoc`, MSHV `2A202603175`) làm việc cùng trợ lý AI (Antigravity): toàn bộ lượt chạy model (`r01`–`r11`), dựng nền và Rework Vòng 2 qua công cụ vẽ/crop/lưới cục bộ (`work/tools/`), chạy script thống kê/QC (`review_stats.py`, `build_release.py`, `qc_sample.py`, `qc_report.py`, `check_submission.py`) và kiểm trực quan từng ảnh đều có AI hỗ trợ thực thi; các nhãn sạch cũng được nạp vào CVAT local để học viên xem và kiểm chứng trực tiếp trên giao diện web.
- **Gold có bị nhiễm không:** **Có.** Theo yêu cầu ban đầu của học viên (*"4 ảnh gold cx pre-label hoặc chạy model luôn đi cứ ghi rõ vào report là được"*), `gold_manual.json` được dựng từ nền `YOLO26m` (`r01`) + `YOLOE-26s` (`r02`) gộp lại rồi kiểm sửa bằng mắt, nên số đo precision/recall của máy trên `gold` (`85.1%` / `94.0%`) bị thiên lệch lạc quan kép (vừa chọn ngưỡng trên `gold`, vừa dùng pre-label làm nền). Ở Vòng 2 giữ nguyên `gold_manual.json` và khai báo trung thực tại đây.
- **Máy / GPU / CVAT dùng; vướng mắc cài đặt:** Windows, GPU `NVIDIA GeForce RTX 4050 Laptop GPU` (6 GB VRAM), `.venv` Python `3.12.16` (tạo bằng `uv`), `torch 2.6.0+cu124`, `ultralytics 8.4.215`, `transformers 5.23.0`. Đã cài Docker và chạy **CVAT `2.74.1` bản chuẩn Docker local tại `http://localhost:8080`** (hơi lệch nhẹ so với bản 2.81 của hướng dẫn; không cài Nuclio/SAM serverless bên trong CVAT; chạy YOLO26/YOLOE/SAM bên ngoài rồi nạp COCO qua API: Project `day08-2A202603175` `id=3`, gồm `gold-manual` #10 [4 ảnh, 68 shapes], `batch-review` #11 [24 ảnh, 532 shapes], `batch-prelabel` #12 [24 ảnh, 570 shapes]).
- **Dùng `data/cache/` thay vì tự chạy?** **Không.** Toàn bộ model được tải mới vào `models/` và chạy trực tiếp trên GPU CUDA (`r01`–`r11`).
- **Lỗi / bug script đã gặp:** Không có bug trong script của đề. Chỉ lưu ý trên Windows cần đặt `PYTHONIOENCODING=utf-8` khi xuất console tiếng Việt và đọc JSON với `utf-8-sig` nếu sửa bằng PowerShell.
- **Một điều làm khác lần sau:** Không duyệt Vòng 1 chỉ trên ảnh thu nhỏ `1.5×` mà áp dụng ngay quy trình zoom + lưới ngay từ đầu, và gán tay sạch 4 ảnh `gold` trên CVAT trước khi chạy model.
