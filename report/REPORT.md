# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy: 11/09/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn: Không** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
(468, "cab", 1, 0.510915, "ImageNet-1K")
- Record này mô tả toàn ảnh như thế nào?
Record này là kết quả dự đoán hạng 1 từ một mô hình phân loại ảnh đã được huấn luyện trên bộ nhãn ImageNet-1K. Nó cho biết: trong số 1000 nhãn có thể có, nhãn mà mô hình cho là khớp nhất với toàn bộ bức ảnh là lớp có class_id = 468, tên là "cab" (xe taxi), với độ tin cậy (score) khoảng 0.51 — tức mô hình gán khoảng 51% xác suất cho khả năng đây là ảnh một chiếc taxi

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Danh sách 1000 lớp là do ImageNet/ILSVRC (dựa trên WordNet) định nghĩa

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
class_id là số nguyên nội bộ model dùng để tính toán, ổn định, không phụ thuộc ngôn ngữ. class_name là nhãn người đọc hiểu được, nhưng chuỗi ký tự dễ trùng/nhập nhằng giữa các bộ dữ liệu khác nhau. taxonomy_name cho biết ID và tên đó thuộc hệ quy chiếu nào — vì cùng một class_id = 468 ở một taxonomy khác (VD: Places365, hoặc một class list tuỳ chỉnh) có thể trỏ tới một khái niệm hoàn toàn khác.
Giữ cả ba đảm bảo kết quả có thể tái lập và tra cứu chính xác, tránh nhầm lẫn khi so sánh/kết hợp dữ liệu từ nhiều model hoặc nhiều taxonomy khác nhau.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Thứ nhất, tiêu chí chọn "chủ thể chính" khi có nhiều đối tượng — ví dụ ưu tiên đối tượng chiếm diện tích lớn nhất, nằm giữa khung hình, hoặc là tiêu điểm lấy nét, vì phân loại toàn ảnh (image classification) về bản chất chỉ gán được một nhãn.
Thứ hai, có cho phép lưu nhiều nhãn không (dùng top-k thay vì chỉ rank 1), và nếu có thì lấy bao nhiêu rank (k=3, k=5...) cùng ngưỡng score tối thiểu để một nhãn được coi là hợp lệ.
Thứ ba, cách xử lý khi các đối tượng thuộc các lớp không cùng mức độ trừu tượng hoặc chồng chéo ngữ nghĩa (VD ảnh có cả "taxi" và "đường phố" — ưu tiên đối tượng cụ thể hơn hay bối cảnh chung hơn).
Thứ tư, nếu mục tiêu thực sự là mô tả đầy đủ các chủ thể, guideline nên nêu rõ classification không phù hợp và cần chuyển sang tác vụ khác (object detection hoặc multi-label classification) thay vì cố ép vào một nhãn duy nhất.

- Vì sao model score không phải ground truth?
Vì score chỉ là ước lượng xác suất chủ quan của model dựa trên những gì nó học được từ dữ liệu huấn luyện, không phải nhãn thực tế đã được xác nhận độc lập (ground truth thường đến từ con người gán nhãn hoặc một nguồn đối chiếu đáng tin cậy khác).
Model có thể sai do nhiều nguyên nhân: dữ liệu huấn luyện thiên lệch hoặc không bao phủ hết trường hợp thực tế, ảnh đầu vào khác phân phối với dữ liệu huấn luyện (domain shift), model có giới hạn kiến trúc (như bản "nano" nhẹ, độ chính xác thấp hơn bản lớn), hoặc bản thân ảnh mơ hồ/có nhiều chủ thể khiến việc gán một nhãn duy nhất vốn không chắc chắn.
Vì vậy score chỉ nên hiểu là "mức độ tự tin của model", cần được đối chiếu/kiểm chứng với ground truth (nếu có) trước khi dùng để đánh giá độ chính xác hay đưa vào làm dữ liệu tin cậy cho các bước sau.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
("cup", 0.381542,  [
      119.95,
      272.2,
      141.78,
      304.16
    ], 21.84, 31.96)
- Diễn giải vị trí box bằng lời:
Diễn giải: khung bao nằm ở vùng phía dưới, lệch về bên trái của ảnh.
Theo trục ngang: x từ 119.95 đến 141.78 (trên tổng chiều rộng 640px), tức khoảng 19–22% từ mép trái — nằm trong một phần ba bên trái của ảnh, không chạm mép.
Theo trục dọc: y từ 272.2 đến 304.16 (trên tổng chiều cao 427px), tức khoảng 64–71% từ mép trên — nằm ở phần ba dưới cùng của ảnh (gần ranh giới giữa vùng giữa và vùng dưới), không chạm đáy.
Kích thước khung nhỏ (21.84 x 31.96 px), chỉ chiếm khoảng 3–7% chiều rộng/chiều cao ảnh — cho thấy đây là một vật thể nhỏ, nằm ở góc dưới-trái của khung hình, không phải chủ thể chiếm phần lớn diện tích ảnh.

- So sánh số prediction ở hai threshold:
threshold=0.20: 17 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']
threshold=0.35: 11 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl']

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
Nếu hạ thấp threshold (VD từ 0.35 xuống thấp hơn): độ bao phủ (recall) tăng — giữ lại được nhiều đối tượng thật hơn, kể cả những cái model không chắc chắn — nhưng đồng thời số lượng prediction giữ lại tăng lên (bao gồm cả false positive), nên khối lượng mà reviewer cần xem/kiểm tra thủ công tăng theo.
Ngược lại, nếu nâng threshold cao hơn: số lượng prediction giảm, chỉ giữ lại những dự đoán model tự tin nhất, giúp giảm khối lượng review — nhưng đánh đổi là độ bao phủ giảm, có nguy cơ bỏ sót (miss) những đối tượng thật mà model chỉ tự tin ở mức trung bình/thấp.

- Đề xuất một quy tắc box chặt:
Khung bao phải áp sát rìa ngoài cùng của phần vật thể nhìn thấy được ở cả 4 cạnh (trên, dưới, trái, phải) — không chừa khoảng đệm (padding) dư quanh vật thể, và không được cắt mất bất kỳ pixel nào thuộc vật thể.
Nếu vật thể bị che khuất một phần bởi vật khác, box chỉ bao phần nhìn thấy được (visible extent), không suy đoán/vẽ thêm cho phần bị che (trừ khi guideline quy định rõ dùng amodal box — cần nêu rõ chọn convention nào).
Nếu vật thể bị cắt bởi rìa ảnh (truncated), cạnh box tương ứng trùng với mép ảnh.
Không bao gồm bóng đổ, phản chiếu, hoặc vật thể liền kề/nền xung quanh vào trong box.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Ngưỡng tối thiểu để vẫn gán nhãn: vật thể bị che/cắt còn lại bao nhiêu % diện tích nhìn thấy thì vẫn đủ để annotate (VD ≥10–20%), dưới ngưỡng đó thì bỏ qua, không gán.
Quy ước box: chỉ bao phần nhìn thấy (visible box) hay ước lượng cả phần bị che (amodal box) — cần chọn một convention thống nhất và áp dụng nhất quán toàn bộ dataset.
Cờ đánh dấu trạng thái: có trường riêng ghi nhận occluded / truncated (và mức độ, nếu cần) để phân biệt với object nhìn thấy đầy đủ, phục vụ phân tích/lọc sau này.
Xử lý khi lớp không xác định được rõ do bị che: nếu phần còn nhìn thấy không đủ đặc trưng để phân biệt lớp, guideline cần quy định ngưỡng tự tin tối thiểu, và nếu dưới ngưỡng đó thì escalate lên reviewer/chuyên gia thay vì để annotator tự đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
("kitchen-006", "bowl", 0.528202, 84, [
        29.0,
        303.0
      ])

- Polygon bổ sung chi tiết gì so với box?
Polygon (segmentation mask dạng đa giác) bổ sung thông tin về hình dạng chính xác của vật thể theo từng pixel/đường viền, thay vì chỉ một hình chữ nhật bao quanh như box

- `instance_id` dùng để làm gì và không phải loại ID nào?
instance_id dùng để phân biệt từng cá thể vật thể riêng lẻ trong ảnh (hoặc xuyên suốt các frame nếu là video), dù cùng chung một class_name/class_id. Nó cũng thường dùng để liên kết các thông tin khác nhau cùng thuộc về một vật thể duy nhất, hoặc để theo dõi (tracking) cùng một vật thể qua nhiều frame video.
Nó không phải là:
class_id — không định danh loại/danh mục vật thể (VD "cup" là gì), mà định danh cá thể cụ thể nào trong số các vật thể cùng loại.
sample_id/coco_image_id — không định danh tấm ảnh, mà định danh một đối tượng bên trong ảnh đó.
Một ID toàn cục nhất quán qua nhiều ảnh/dataset khác nhau (trừ khi hệ thống có thiết kế riêng cho việc đó, như bài toán re-identification) — mặc định instance_id thường chỉ có ý nghĩa cục bộ trong phạm vi một ảnh hoặc một track, không đảm bảo cùng một vật thể thực tế sẽ giữ cùng instance_id nếu xuất hiện lại ở ảnh khác.

- Đề xuất một quy tắc biên mask:
Đường biên phải bám sát theo đúng ranh giới thị giác thực tế của vật thể (nơi màu sắc/kết cấu chuyển từ vật thể sang nền), với sai số cho phép tối đa khoảng 1–2 pixel so với biên thật — không đơn giản hóa (simplify) quá mức làm mất chi tiết góc cạnh, cũng không vẽ quá nhiều điểm dư thừa không cần thiết.
Với các cạnh mờ/khó xác định (chuyển màu dần, vật trong suốt, vật có viền mảnh như tóc/lông): quy ước lấy điểm giữa của vùng chuyển tiếp làm biên, áp dụng nhất quán cho mọi trường hợp tương tự.
Nếu vật thể có lỗ hổng thực sự để lộ nền phía sau (VD tay cầm cốc, khung xe đạp): mask phải thể hiện đúng lỗ đó (dùng đa giác có vòng trong/interior ring), không được tô đặc lấp đầy.
Với vật thể bị che khuất: biên mask chỉ vẽ theo phần nhìn thấy được (visible mask), dừng lại đúng tại ranh giới với vật che, trừ khi guideline chọn dùng amodal mask.
Với hai vật thể liền kề/chạm nhau: biên của chúng phải khớp sát nhau, không chồng lấn (không có pixel được gán cho cả hai) và không để hở khoảng trống giữa hai mask.
Cần một tiêu chí định lượng để QA: VD IoU giữa mask được vẽ và mask tham chiếu/chuẩn phải đạt ngưỡng tối thiểu (như ≥0.9) mới được chấp nhận

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Ngưỡng để vẫn annotate: vùng bị che/mờ đến mức nào thì vẫn đủ tin cậy để vẽ mask, và dưới ngưỡng đó thì bỏ qua thay vì đoán mò.
Quy ước biên khi mờ/không rõ ràng: vị trí chính xác của đường biên khi không có ranh giới sắc nét (dùng điểm giữa vùng chuyển tiếp hay quy tắc khác) — cần thống nhất một lần cho toàn dataset, không để từng annotator tự chọn.
Tách hay gộp khi hai vật thể chạm/che nhau quá nhiều: nếu ranh giới giữa hai vật thể liền kề thực sự không thể phân biệt (do che khuất gần như hoàn toàn), có tách thành hai mask riêng theo suy đoán hay gộp/đánh dấu là một nhóm — quyết định này ảnh hưởng lớn đến chất lượng dữ liệu nên cần escalate.
Amodal hay chỉ visible: khi vật thể bị che, có ước lượng phần bị che (amodal) hay chỉ vẽ phần thấy được — nếu annotator không chắc chắn thuộc trường hợp nào, cần escalate lên reviewer.
Mức độ tự tin về lớp khi vùng nhìn thấy quá ít: giống với trường hợp box, nếu phần còn lại không đủ đặc trưng để xác định lớp, cần ngưỡng tối thiểu và cơ chế escalate lên chuyên gia thay vì annotator tự quyết.


## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn duy nhất cho toàn ảnh (class_id + class_name theo một taxonomy cố định) | Ảnh có nhiều chủ thể nên không rõ chọn nhãn nào là chính; ranh giới mơ hồ giữa các lớp gần nghĩa nhau; ảnh không khớp lớp nào trong taxonomy | Áp dụng quy tắc chọn chủ thể chính (diện tích lớn nhất/ở giữa khung hình...); nếu mơ hồ giữa hai lớp gần, chọn theo hướng dẫn phân biệt cụ thể hoặc gắn cờ nghi vấn; đánh dấu "không xác định" nếu ngoài phạm vi taxonomy | Nhãn có đúng quy tắc chọn chủ thể chính không; các ca ảnh nhiều chủ thể/mơ hồ có được xử lý nhất quán giữa các annotator không |
| Phát hiện vật thể | Bounding box theo từng instance (bbox_xyxy/xywh) + class_id + instance_id | Box không chặt (dư/thiếu vùng); vật thể bị che khuất hoặc cắt mép chưa rõ quy ước visible/amodal; nhiều vật thể chồng lấn khó tách box; threshold ảnh hưởng recall vs khối lượng review | Vẽ box áp sát rìa vật thể theo quy tắc "box chặt"; áp dụng đúng quy ước occlusion/truncation đã định; gán instance_id riêng cho mỗi cá thể, kể cả cùng lớp | Độ chặt của box (so IoU với box chuẩn); occlusion/truncation có xử lý nhất quán không; có bỏ sót hoặc trùng lặp instance không; lớp gán có đúng không |
| Instance segmentation | Polygon/mask theo từng instance (class_id + instance_id, có thể kèm interior ring cho lỗ hổng) | Biên mờ khó xác định chính xác; vật thể chạm/che nhau khó tách biên; lỗ hổng thực trong vật thể dễ bị tô đặc nhầm; cần quyết định visible hay amodal | Vẽ biên trong dung sai cho phép (VD ≤1–2px); dùng quy ước điểm giữa vùng chuyển tiếp cho biên mờ; vẽ đúng lỗ hổng bằng interior ring; chỉ mask phần nhìn thấy khi bị che theo quy ước đã chọn | IoU giữa mask vẽ và mask chuẩn; biên giữa các instance liền kề có chồng lấn hoặc hở khoảng trống không; lỗ hổng và occlusion có xử lý đúng quy ước không |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
Mọi thông tin định danh cá nhân xuất hiện trong ảnh thô phải được phát hiện và làm mờ/che (redact) trước khi ảnh được đưa cho annotator xem hoặc lưu vào tập dữ liệu huấn luyện — annotator không bao giờ được tiếp cận bản gốc chưa xử lý.
Việc truy cập ảnh gốc chưa redact (nếu cần giữ để phục vụ mục đích khác) phải giới hạn ở số lượng người tối thiểu, có ghi log truy cập, và tách biệt hoàn toàn khỏi luồng annotate/huấn luyện thông thường.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
giảng viên hướng dẫn trước khi quyết định bước tiếp theo

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
