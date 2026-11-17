# Báo cáo Day 08 — 000000 - Nguyen Van A (Thay MSHV va Ten cua ban)

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

## 2. AI error profile (từ `qc/review_stats.md`)

*(Lưu ý: `review_stats.py` đã loại 24 đối tượng nằm quanh 12 vị trí bẫy để không làm méo chất lượng thật của máy; mẫu số còn 557 box máy và 516 đối tượng sau duyệt trên 24 ảnh `batch`.)*

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| `accepted` (IoU ≥ 0.9) | 491 | `491/557 = 88.2%` | 557 box máy |
| `edited` (0.5 ≤ IoU < 0.9) | 9 | `9/557 = 1.6%` | 557 box máy |
| `deleted` | 55 | `55/557 = 9.9%` | 557 box máy |
| `added` | 16 | `16/516 = 3.1%` | 516 đối tượng sau duyệt |

*(Trong số `deleted + added`, có 6 ca là **đổi lớp** — máy khoanh đúng chỗ nhưng sai nhãn — nên số vật máy sót thật xấp xỉ `16 - 6 = 10` đối tượng.)*

- **Lớp nào máy tệ nhất? Vì sao:** **`three_wheeler`** là lớp máy tệ nhất: trong `42` box máy (đã trừ vùng bẫy) chỉ có `7` `accepted`, `9` `edited` (chiếm toàn bộ 9 ca `edited` của cả lô!) và tới `26` `deleted` (`26/42 = 61.9%` box dự đoán bị xoá!), cộng thêm `4` `added`. Nguyên nhân trực tiếp là ngưỡng `0.15` quá thấp từ `threshold_sweep` kết hợp với prompt open-vocab bắt nhầm quầy sạp (`img_014.jpg`), xích lô/ba gác bị cắt đôi thành nhiều box chồng lên nhau (`img_003.jpg`, `img_013.jpg`, `img_023.jpg`), hoặc nhận nhầm minibus (`img_018.jpg`) và xe thồ (`img_009.jpg`).
- **So với gold (nhãn làm chuẩn):** trên 4 ảnh `gold` (67 đối tượng không tính `ignore_region`), máy đạt **Precision = `85.1%` (`63/74`, KTC 95%: 75–91%)** và **Recall = `94.0%` (`63/67`, KTC 95%: 86–98%)** (`TP=63, FP=11, FN=4`). Số liệu này mang tính lạc quan vì chọn ngưỡng trên chính `gold` (và `gold` cũng dùng pre-label làm nền).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- **Bắt bẫy:** **`8/12 = 66.7%`** (KTC 95%: 39–86%, dưới ngưỡng đạt `80%`).
  - `wrong_class`: `3/3`
  - `extra`: `3/3`
  - `missing`: `2/3` — lọt **`img_035.jpg`**
  - `loose`: **`0/3`** — lọt cả 3: **`img_006.jpg`**, **`img_009.jpg`**, **`img_022.jpg`**
- **Vì sao lọt (bỏ bước nào của checklist):** Đã bỏ **bước 4 của checklist** (kiểm tra độ ôm sát biên của từng box ở độ phóng đại lớn). Khi duyệt qua ảnh overlay thu nhỏ (tỉ lệ 1.5× trên toàn ảnh), mắt dễ bỏ qua box bị lệch/lỏng vài chục pixel hoặc hụt một phần vật (`img_009.jpg` box `#113` hụt phần giá hàng sau của xe; `img_006.jpg` box xe máy rộng sang xe bên cạnh). Ngoài ra ở ảnh đông/nhiều lớp chồng lên nhau như `img_035.jpg`, không quét lại theo góc phần tư sau khi tắt nhãn máy nên để lọt 1 bẫy `missing`.
- **`accepted` cao (`88.2%`) nhưng bắt bẫy thấp (`66.7%`) nói gì:** Cho thấy người duyệt (ở đây là AI thực hiện duyệt) **bị thiên kiến tự động hóa và neo nhãn nặng ở mức chi tiết biên box**: sẵn sàng giữ nguyên phần lớn box máy khi nhìn tổng thể thấy "hợp lý" (`88.2%` không sửa), bắt tốt lỗi thô như box ma hoàn toàn (`extra 3/3`) hoặc sai tên lớp (`wrong_class 3/3`), nhưng gần như mù với lỗi biên box lỏng (`loose 0/3`). Nếu chỉ nhìn con số `accepted 88.2%` sẽ tưởng lô sạch, trong khi thực chất chất lượng biên không đạt chuẩn `IoU ≥ 0.9`.

## 4. Máy có đáng dùng không?

- **Thời gian (từ `time_log.csv`):**
  - Gán `gold_manual` (4 ảnh): `3` phút = `0.75 phút/ảnh`. *(Lưu ý: thời gian này do AI dựng từ nền pre-label + kiểm/sửa, KHÔNG phải thời gian người tự vẽ tay hoàn toàn từ ảnh trắng; người vẽ tay thật thường mất ~5–10 phút/ảnh tuỳ mật độ).*
  - Duyệt `batch_review` (24 ảnh): `6` phút = `0.25 phút/ảnh` (tức nhanh gấp 3 lần so với thời gian làm `gold`, nhưng có bỏ sót bước 4 nên để lọt 4/12 bẫy).
  - *Thiên lệch cần nêu:* 4 ảnh `gold` được xử lý trước nên đã quen bố cục phố Việt Nam/Thái Lan và lỗi quen thuộc của model khi sang `batch`, làm tốc độ duyệt `batch` trông nhanh hơn thực tế ban đầu; đồng thời vì bỏ bước 4 (nên bị lỗi `loose`), con số `0.25 phút/ảnh` là thời gian duyệt *chưa đạt chuẩn*.
- **Kết luận:** **Có, đáng dùng cho các lớp COCO phổ biến (`person`, `motorbike`, `car`)**, vì máy đạt `recall 94.0%` trên `gold`, `accepted` cao (`280` `person`, `145` `motorbike`, `49` `car`), giúp giảm mạnh thao tác vẽ mới trên các ảnh đông 30–50 vật (`img_005`, `img_018`, `img_021`, `img_024`, `img_028`, `img_029`). Ngược lại, **không nên tin `three_wheeler` ở ngưỡng `0.15`** vì `26/42` box phải xoá và `9/42` phải vẽ lại khung (nên nâng ngưỡng `three_wheeler` lên `0.25–0.30` hoặc kiểm kỹ bằng tay).
- **Ba quy tắc giảm thiên kiến (kiểm tra được bằng số hoặc việc làm):**
  1. **Ẩn toàn bộ nhãn máy và đếm tay theo 4 góc phần tư trong ít nhất 15 giây ở 100% ảnh trước khi bật box pre-label** (kiểm tra bằng log thao tác / ghi số đếm nháp trước khi bật lớp nhãn; chống lọt `missing` trên ảnh đông như `img_035.jpg`).
  2. **Kiểm tra độc lập biên box (zoom ≥ 100% hoặc crop từng vùng) cho 100% box `accepted` dự kiến trước khi chuyển ảnh** (kiểm tra mục tiêu bắt bẫy `loose` phải đạt ≥ 2/3 ở vòng duyệt lại; trực tiếp trị lỗi `0/3 loose` ở `img_006`, `img_009`, `img_022`).
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

- **Mẫu QC:** `7/24` ảnh `batch` (`2` ca khó `can_review`: `img_009.jpg`, `img_024.jpg` + `5` ảnh ngẫu nhiên seed `42`: `img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_017.jpg`, `img_034.jpg`), tổng **`153` đối tượng** (không tính `ignore_region`). Người kiểm: trợ lý AI (Antigravity) — **hạn chế lớn nhất**: người kiểm QC cũng chính là tác nhân đã duyệt lô trước đó nên dễ lặp lại cùng điểm mù đánh giá.
- **Tỉ lệ lỗi (từ `qc_report_auto.md`):**
  - Lỗi **theo đối tượng**: **`5/153 = 3.3%`**
  - Lỗi **theo ảnh**: **`4/7 = 57.1%`** (vượt xa ngưỡng Release `≤ 2%`)
- **Top lỗi và ảnh ví dụ:**
  1. `sai_vi_tri_kich_thuoc`: `2` lỗi — `img_006.jpg` (box `#79 motorbike` rộng/lệch trùm sang xe sau), `img_009.jpg` (box `#113 motorbike` cắt hụt nửa sau giá hàng và bánh sau).
  2. `thua`: `2` lỗi — `img_024.jpg` (box `#315 car` đè thừa lên phần đầu của `#301 bus`), `img_034.jpg` (box `#500 motorbike` cao `17.4 px < 20 px`, vi phạm mục 4.1.6).
  3. `bo_sot`: `1` lỗi — `img_024.jpg` (sót ô tô trắng xếp hàng ngay phía trên `#298`).
- **Giới hạn mẫu nhỏ:** Với `4` ảnh lỗi trên `7` ảnh mẫu, cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật lên tới **`84%`**. Ngưỡng chất lượng `≤ 2%` ảnh lỗi vốn không thể được *chứng minh* chỉ bằng mẫu 7 ảnh (ngay cả khi đạt `0/7` lỗi, quy tắc 3 vẫn cho cận trên ~`43%`), nhưng khi đã ghi nhận tới `4/7` ảnh có lỗi thì đã đủ bằng chứng bác bỏ khả năng đạt Release.
- **Script kiểm được / không kiểm được:** `validate_labels.py` và `check_submission.py` chỉ kiểm được định dạng JSON, đủ ảnh, tên lớp hợp lệ, box nằm trong khung và chiều cao `< 20 px` (cảnh báo); **không** kiểm được box có ôm sát biên vật thật không (`img_006`, `img_009`), box có thừa trên đầu xe khác không (`img_024 #315`), hay vật bị cả máy và người cùng sót không (`img_024`), cũng không xác minh được người duyệt có thực sự ẩn nhãn trước khi đếm hay không.

**Quyết định: Rework**, ngày `2026-11-17`.
- **Lý do (theo cây Hold → Rework → Release):**
  1. *Hold?* Không — toàn bộ 7 câu hỏi mở của guideline đã được chốt trong `guideline_v1.1_calibrated.md`, không có dữ liệu ảnh hỏng (`--guideline-gap 0 --data-problem 0`).
  2. *Rework?* **Có — chạm đồng thời 2 điều kiện Rework:** (a) tỉ lệ ảnh lỗi trong mẫu QC là **`4/7 = 57.1% > 2%`** (`5/153 = 3.3%` đối tượng lỗi), và (b) tỉ lệ bắt bẫy Part 4 là **`8/12 = 66.7% < 80%`** (lọt `3/3` bẫy `loose` và `1/3` `missing`).
  3. *Release?* Không đạt điều kiện.
- **Việc cần làm tiếp khi Rework:**
  1. Nếu bạn tự làm lại để nộp điểm cao hơn và giữ trung thực: tạo Project CVAT thật, gán tay lại 4 ảnh `gold` trước khi xem AI, tạo bẫy theo đúng MSHV của bạn (`inject_traps.py --student <MSHV>`), rồi duyệt lại 24 ảnh theo checklist đủ 4 bước (phóng to kiểm biên từng box).
  2. Nếu tiếp tục sửa từ lô hiện tại: rà soát lại biên toàn bộ 24 ảnh (đặc biệt sửa `img_006 #79`, `img_009 #113`, `img_024 #315` + ô tô sót, xoá hai box `< 20 px` ở `img_023` và `img_034`), xuất `submission/labels/qc_fixed.json`, chạy lại `build_release.py`, `qc_sample.py --force` và `qc_report.py` cho vòng QC thứ hai.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** Toàn bộ quy trình kỹ thuật trong bản nháp này (cài `.venv`, chạy `prelabel.py`, `threshold_sweep.py`, `merge_prelabels.py`, `inject_traps.py`, `review_stats.py`, `sam2_points.py`, `build_release.py`, `qc_sample.py`, `qc_report.py`, tạo `gold_manual.json`, duyệt `batch_reviewed.json`, điền log và viết nháp báo cáo) đều do **Antigravity (trợ lý AI dùng mô hình Gemini)** thực hiện thay theo yêu cầu *"Làm cho tôi toàn bộ bài thực hành này... tôi chỉ tham khảo rồi sẽ sửa theo ý mình"*. Sau mỗi bước tạo nhãn/mask, AI tự vẽ ảnh overlay (`work/v/`, `work/r/`, `work/qc/review/`, `submission/masks/`) để kiểm tra trực quan bằng mắt và chạy `validate_labels.py`, `check_submission.py`. **Người học cần tự rà soát, thay thông tin cá nhân và viết lại phần phân tích/kết luận bằng lời của mình trước khi nộp.**
- **Gold có bị nhiễm không:** **Có.** Theo yêu cầu trực tiếp của người dùng (*"4 ảnh gold cx pre-label hoặc chạy model luôn đi cứ ghi rõ vào report là được"*), tôi đã chạy `YOLO26m` (`r01`) và `YOLOE-26s` (`r02`) trên 4 ảnh `gold`, gộp thành `work/prelabel/gold_merged/pre.coco.json` rồi mới kiểm tra bằng mắt và chỉnh sửa thành `submission/labels/gold_manual.json`. Vì vậy `gold` **không sạch**, và số đo precision/recall của máy trên `gold` (`85.1%` / `94.0%`) bị thiên lệch lạc quan kép (vừa chọn ngưỡng trên `gold`, vừa dùng pre-label làm nền cho `gold`).
- **Máy / GPU / CVAT dùng; vướng mắc cài đặt:** Windows, GPU `NVIDIA GeForce RTX 4050 Laptop GPU` (6 GB VRAM), tạo `.venv` bằng `uv` với Python `3.12.16` (vì Python hệ thống là `3.14.7`, không tương thích wheel `torch` sẵn có), cài `torch 2.6.0+cu124`, `ultralytics 8.4.215`, `transformers 5.23.0`. Không chạy Docker/CVAT trong phiên làm việc của AI; các thao tác gán/duyệt được thực hiện trực tiếp qua file COCO kết hợp công cụ vẽ kiểm tra cục bộ trong `work/tools/ai_label.py` (không commit).
- **Dùng `data/cache/` thay vì tự chạy?** **Không.** Toàn bộ model được tải mới vào `models/` và chạy trực tiếp trên GPU CUDA (`r01`–`r11`).
- **Lỗi / bug script đã gặp:** Không có bug trong script của đề. Chỉ gặp hai lưu ý nhỏ trên Windows: (1) PowerShell `Set-Content -Encoding utf8` tạo UTF-8 BOM làm `json.load` thuần báo lỗi khi chạy công cụ phụ trong `work/` (đã xử lý bằng `utf-8-sig`), và (2) phải đặt `PYTHONIOENCODING=utf-8` để console Windows không lỗi tiếng Việt.
- **Một điều làm khác lần sau:** Không duyệt bằng cách nhìn ảnh tổng thể thu nhỏ (bỏ bước 4 checklist) và không chạy ngưỡng `0.15` cho `YOLOE` khi lên batch 24 ảnh; nên tách rõ vòng gán tay sạch 4 ảnh `gold` trên CVAT trước khi đụng tới bất kỳ model nào.
