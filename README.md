# ESP32-S3 gesture demo

Pipeline nhận dạng 4 cử chỉ `fist`, `palm`, `peace`, `thumbs_up` và lớp `none` cho tư thế khác/không có tay. Firmware có thể dùng 4 cử chỉ đó để bật/tắt hai LED ngoài. Nếu thư mục ảnh tên `other`, script train cũng dùng nó cho lớp `none`.

## Bắt đầu nhanh cho người mới

Dự án nhận dạng 4 cử chỉ tay (`fist`, `palm`, `peace`, `thumbs_up`) và lớp `none` cho tư thế khác/không có tay. `test_ai.py` chỉ chạy CNN ảnh TFLite. Model thử nghiệm mặc định là `model_raw_focus4.tflite`, được train bằng ảnh grayscale gốc.

### Thử trên laptop

Cài Python 3.8–3.11 và Git, mở PowerShell rồi chạy:

```powershell
git clone https://github.com/UnknownLight2/Esp32-gesture.git
cd Esp32-gesture
py -m pip install --upgrade "pip<25.1"
py -m pip install -r requirements-collect.txt -r requirements-train.txt
py test_ai.py --camera 0
```

Đưa bàn tay vào giữa khung hình; nhấn `Q` hoặc `Esc` để thoát. Nếu laptop có nhiều camera, thử `--camera 1`. Kiểm tra ảnh tĩnh bằng `py test_ai.py --image "C:\duong\dan\anh.jpg"`; đánh giá toàn bộ tập test bằng `py test_ai.py --test-dir test_images`.

### Thử bằng board ESP32-S3

1. Cài Arduino IDE, ESP32 board support và thư viện `Chirale_TensorFlowLite` trong Library Manager.
2. Mở `esp32s3_gesture.ino`, chọn **ESP32S3 Dev Module**. Với board N16R8, bật PSRAM theo cấu hình board và chọn partition phù hợp.
3. Nạp sketch. Kết nối máy tính vào Wi-Fi `ESP32-GESTURE` (mật khẩu `12345678`), rồi mở `http://192.168.4.1` để xem camera và kết quả nhận dạng.
4. Muốn thử nhận dạng video ESP32 trên laptop, cài dependencies như phần trên, kết nối vào Wi-Fi của board và chạy `py test_ai.py --url http://192.168.4.1:81/stream`.

Firmware ưu tiên `model_raw_focus4.h`; camera và model chạy trên ESP32-S3, còn trang web chỉ hiển thị video/trạng thái.

### Chức năng chính

- Xem video camera ESP32-S3 trên trang web nội bộ và đọc trạng thái qua `/status`.
- Thu ảnh có nhãn từ webcam laptop hoặc stream của ESP32 để bổ sung dữ liệu.
- Huấn luyện CNN ảnh thành TFLite, kiểm tra trên tập ảnh có nhãn, webcam hoặc stream ESP32.

Firmware có thể dùng cử chỉ để điều khiển hai LED ngoài; ánh xạ và chân nối nằm ở mục kiểm tra trên ESP32. Độ tin cậy phụ thuộc dữ liệu và camera.

Bản báo cáo demo và poster trình bày nhóm được lưu trong `reports/`; có thể mở trực tiếp sau khi tải repository.

## 1. Trước hết: camera và IP

1. Mở `esp32s3_gesture.ino` trong Arduino IDE, chọn **ESP32S3 Dev Module**, bật PSRAM theo hướng dẫn board (N16R8 thường là OPI PSRAM), USB CDC On Boot bật nếu dùng cổng USB native, và chọn partition phù hợp.
2. Firmware mặc định tạo Wi-Fi AP `ESP32-GESTURE` (mật khẩu `12345678`) và đặt IP `192.168.4.1`. Kết nối laptop vào AP này rồi mở `http://192.168.4.1`. Không cần router; internet laptop có thể tạm ngắt khi đang nối AP.
3. Nạp sketch trước khi thu mẫu. Serial Monitor ở 115200 sẽ in AP/IP; trang `/` xem camera, `/status` đọc JSON, `http://192.168.4.1:81/stream` là MJPEG.

## 2. Thu sample

Cài Python 3.8-3.11 và dependency. Nếu pip quá cũ, nâng pip lên bản cuối hỗ trợ Python 3.8 trước:

```powershell
py -m pip install --upgrade "pip<25.1"
py -m pip install -r requirements-collect.txt
```

Có thể lấy ảnh ngay bằng webcam laptop, chưa cần nạp firmware ESP32. Mỗi lệnh mở webcam index `0`; nhấn SPACE để lưu ảnh vào `dataset/<nhãn>/`, Q để chuyển sang nhãn kế tiếp. Nếu máy có nhiều camera, thử `--camera 1`. `--mirror-preview` chỉ lật hình xem trước, ảnh lưu vẫn giữ chiều gốc.

```powershell
py collect_data.py --camera 0 --label fist
py collect_data.py --camera 0 --label palm
py collect_data.py --camera 0 --label peace
py collect_data.py --camera 0 --label thumbs_up
py collect_data.py --camera 0 --label none
```

Hoặc thu trực tiếp bằng camera ESP32 (sau khi nạp firmware stream và nối laptop vào AP `ESP32-GESTURE`):

```powershell
py collect_data.py --url http://192.168.4.1:81/stream --label fist
py collect_data.py --url http://192.168.4.1:81/stream --label palm
py collect_data.py --url http://192.168.4.1:81/stream --label peace
py collect_data.py --url http://192.168.4.1:81/stream --label thumbs_up
py collect_data.py --url http://192.168.4.1:81/stream --label none
```

Nhấn SPACE để lưu frame, Q để thoát. Thu tối thiểu vài trăm ảnh mỗi lớp, thay đổi người, khoảng cách, hướng tay, ánh sáng và nền. `none` nên có nhiều tư thế tay khác nhau và cả khung không có tay. Không chia frame gần như trùng nhau vào cả train và validation; script hiện chia ngẫu nhiên ảnh, nên accuracy validation ban đầu có thể lạc quan.

Ảnh webcam laptop dùng được để train thử trước khi nạp model. Vì M21-45 và webcam laptop có thể khác ống kính, màu và góc nhìn, nên sau đó hãy thu thêm ảnh từ ESP32 và train lại để model khớp với camera thực tế.

## 3. Train và xuất model

Tạo thêm 100 biến thể ảnh cho mỗi lớp và giữ nguyên `dataset` gốc:

```powershell
py augment_dataset.py --data dataset --out dataset_augmented --per-class 100
```

Các biến thể được tạo từ ảnh hiện có, không thay thế ảnh chụp mới. Khi huấn luyện, ảnh gốc và các biến thể liên quan được giữ cùng một nhóm; các ảnh thu liên tiếp trong cửa sổ 5 giây cũng được gom nhóm để giảm rò rỉ khung hình gần giống sang validation.

```powershell
py -m pip install -r requirements-train.txt
py train_model.py --data dataset_augmented --epochs 25 --out model_outline.h
```

Script xuất file `.h` và `.tflite` theo tên `--out`. Mặc định model dùng Sobel nhẹ để nhấn nét cạnh. Để train trực tiếp trên ảnh grayscale mà không thêm viền, chạy với `--no-outline`, rồi khi test desktop cũng thêm `--no-outline`:

```powershell
py train_model.py --data dataset_train_complete --epochs 25 --seed 43 --no-outline --class-multiplier thumbs_up=4 --class-multiplier none=4 --out model_raw_focus4.h
py test_ai.py --test-dir test_images --model model_raw_focus4.tflite
```

Sketch hiện ưu tiên `model_raw_focus4.h` và bỏ outline khi dùng header này. Tệp đã nằm cạnh `.ino`; mở sketch trong Arduino IDE, chọn **ESP32S3 Dev Module**, bật PSRAM (OPI PSRAM cho N16R8), chọn partition phù hợp, cài `Chirale_TensorFlowLite` nếu thiếu rồi bấm **Upload**. LED1 nối GPIO1, LED2 nối GPIO2 như sơ đồ ở mục 5.

## 4. Huấn luyện và kiểm tra CNN ảnh

Model thử nghiệm hiện tại dùng 7.663 ảnh trong `dataset_train_complete` (ảnh tại chỗ, ảnh augmentation, ảnh MIT và HaGRID train). Thư mục huấn luyện tổng hợp và ZIP HaGRID không đưa lên GitHub vì dung lượng lớn. Muốn tạo lại bộ train, tải `tinyml_improved_20261016.zip`, đặt cạnh README rồi chạy:

```powershell
py augment_dataset.py --data dataset --out dataset_augmented --per-class 100
py augment_dataset.py --data dataset --out dataset_augmented_v2 --per-class 500
py prepare_cnn_data.py --hagrid-zip tinyml_improved_20261016.zip
```

Lệnh cuối tạo `dataset_train_complete`, `dataset_train_all.zip` và `test_hagrid`. Chỉ phần train của HaGRID được đưa vào dữ liệu huấn luyện; phần test được giữ riêng. `test_images` gồm 25 ảnh trùng chính xác với một phần HaGRID test, nên không tính là một phép thử độc lập thứ hai.

Để huấn luyện lại bằng grayscale gốc, không vẽ đường viền:

```powershell
py train_model.py --data dataset_train_complete --epochs 25 --seed 43 --no-outline --class-multiplier thumbs_up=4 --class-multiplier none=4 --out model_raw_focus4.h
```

Kiểm tra nhanh một ảnh, cả tập test, webcam laptop hoặc stream ESP32:

```powershell
py test_ai.py --image "C:\duong\dan\anh.jpg"
py test_ai.py --test-dir test_images
py test_ai.py --camera 0
py test_ai.py --url http://192.168.4.1:81/stream
```

`test_ai.py` chỉ chạy classifier CNN ảnh; model mặc định là `model_raw_focus4.tflite`. Để kiểm tra model firmware cũ được train với Sobel outline, chỉ định `--model model_outline.tflite --outline`. Khi mở luồng trực tiếp, nhấn `I` để chọn ảnh tĩnh; đóng ảnh thì live tiếp tục.

Lần kiểm thử mới nhất của candidate trên `test_images` đạt 6/25 (24%); trên HaGRID test đạt 20/75 (26,7%). Trên tập người dùng, fist 2/5, palm 2/5, peace 2/5, thumbs_up 0/5 và none 0/5. Dù tổng thể nhích lên so với model grayscale trước, nó chưa sửa được like/none và chưa nên dùng làm model demo cuối.

Để bổ sung dữ liệu đúng camera cho like và khung hình không có tay, thu các mẫu đa dạng (đổi khoảng cách, vị trí, ánh sáng, nền; tránh giữ một tư thế rồi lưu liên tục nhiều frame giống nhau):

```powershell
py collect_data.py --camera 0 --label thumbs_up
py collect_data.py --camera 0 --label none
py prepare_cnn_data.py --force
py train_model.py --data dataset_train_complete --epochs 25 --seed 43 --no-outline --class-multiplier thumbs_up=4 --class-multiplier none=4 --out model_raw_focus4.h
py test_ai.py --test-dir test_images
```

Ứng viên grayscale vẫn yếu trên các ảnh test hiện có; hãy kiểm tra hoạt động thực tế trên board trước khi coi kết quả là đáng tin cậy.

## 5. Kiểm tra nhận dạng trên ESP32

Mở trang IP in ở Serial Monitor. Khi có model, trang hiển thị lớp, score và trạng thái LED; `/status` trả JSON. Nhãn hiển thị dùng ngưỡng `0.55`; lệnh điều khiển LED dùng ngưỡng riêng `0.85` cùng điều kiện ổn định 4 frame.

### Nối 2 LED và ánh xạ cử chỉ

Dùng hai LED ngoài và một điện trở 220–330 Ω nối tiếp cho mỗi LED. GPIO1 và GPIO2 có trên pinout board và không nằm trong mapping camera hiện tại:

| ESP32-S3 | Nối tới |
|---|---|
| GPIO1 | Điện trở → chân dài (anode) LED1 |
| GPIO2 | Điện trở → chân dài (anode) LED2 |
| GND | Chân ngắn (cathode) của cả hai LED |

Firmware xuất mức HIGH để bật LED. Cử chỉ được giữ ổn định ít nhất 4 khung hình và confidence từ 85% mới phát lệnh:

| Cử chỉ | Lệnh |
|---|---|
| fist (nắm đấm) | Tắt LED1 |
| palm (xòe tay) | Bật LED1 |
| peace (2 ngón) | Bật LED2 |
| thumbs_up (like) | Tắt LED2 |
| none / unknown | Không đổi LED |

Giữ nguyên cử chỉ không lặp lệnh; cần rời cử chỉ/nhận `none` hoặc `unknown` vài khung hình trước khi lệnh đó được nhận lại. Trạng thái hai LED xuất hiện trên trang camera và trong `/status`.

**Lưu ý về độ chính xác:** model thử nghiệm hiện tại đạt 0/5 trên `thumbs_up` và 0/5 trên `none` ở tập người dùng. Ngưỡng lệnh 85% ngăn một số dự đoán yếu làm đổi LED, nhưng không sửa được model; vì vậy hai cử chỉ này có thể chưa kích hoạt đúng cho tới khi thu thêm dữ liệu và train lại.

## Giới hạn xác minh

- Pipeline train/test CNN đã chạy bằng Python 3.8 và TensorFlow 2.13.0 trong môi trường hiện tại. Chưa có `arduino-cli`, nên chưa xác nhận build firmware trên máy này.
- Thu bằng webcam laptop không cần ESP32; thu qua URL MJPEG thì cần nạp firmware stream trước.
