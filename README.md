# ESP32-S3 gesture demo

Pipeline tạo 5 nhãn `fist`, `palm`, `peace`, `thumbs_up`, `none`. Lớp `none` là các tư thế bàn tay không thuộc bốn cử chỉ mục tiêu. Nếu thư mục ảnh của bạn tên `other`, script train cũng nhận thư mục đó làm dữ liệu cho lớp `none`. Chưa gán hành động nào cho cử chỉ.

## Bắt đầu nhanh cho người mới

Dự án nhận dạng 4 cử chỉ tay (`fist`, `palm`, `peace`, `thumbs_up`) và lớp `none` cho tư thế khác/không có tay. Bạn có thể thử nhận dạng ngay trên laptop bằng webcam mà chưa cần ESP32; model mẫu `model_outline.tflite` đã có sẵn trong repository.

### Thử trên laptop

Cài Python 3.8–3.11 và Git, mở PowerShell rồi chạy:

```powershell
git clone https://github.com/UnknownLight2/Esp32-gesture.git
cd Esp32-gesture
py -m pip install --upgrade "pip<25.1"
py -m pip install -r requirements-collect.txt -r requirements-train.txt -r requirements-landmark.txt
py test_ai.py --camera 0
```

Đưa bàn tay vào giữa khung hình. Cửa sổ hiển thị bộ xương 21 khớp, nhãn landmark và kết quả CNN; nhấn `Q` hoặc `Esc` để thoát. Classifier landmark mẫu `model_landmark.keras` có sẵn; MediaPipe tự tải detector khi chạy lần đầu. Nếu laptop có nhiều camera, thử `--camera 1`. Có thể kiểm tra một ảnh bằng `py test_ai.py --image "C:\duong\dan\anh.jpg"`.

### Thử bằng board ESP32-S3

1. Cài Arduino IDE, ESP32 board support và thư viện `Chirale_TensorFlowLite` trong Library Manager.
2. Mở `esp32s3_gesture.ino`, chọn **ESP32S3 Dev Module**. Với board N16R8, bật PSRAM theo cấu hình board và chọn partition phù hợp.
3. Nạp sketch. Kết nối máy tính vào Wi-Fi `ESP32-GESTURE` (mật khẩu `12345678`), rồi mở `http://192.168.4.1` để xem camera và kết quả nhận dạng.
4. Muốn thử nhận dạng video ESP32 trên laptop, cài dependencies như phần trên, kết nối vào Wi-Fi của board và chạy `py test_ai.py --url http://192.168.4.1:81/stream`.

Firmware dùng `model_outline.h` có sẵn để nhận dạng trực tiếp. Camera ESP32-S3 và phần nhận dạng trên laptop/board cần được kiểm tra với đúng revision phần cứng; xem các mục bên dưới nếu muốn tự thu dữ liệu hoặc huấn luyện lại.

### Chức năng chính

- Xem video camera ESP32-S3 trên trang web nội bộ và đọc trạng thái qua `/status`.
- Thu ảnh có nhãn từ webcam laptop hoặc stream của ESP32 để bổ sung dữ liệu.
- Huấn luyện model TFLite hoặc classifier landmark riêng, kiểm tra trên ảnh, webcam hoặc video ESP32.
- Vẽ 21 điểm khớp và nhận dạng bằng MediaPipe trên laptop; phần landmark này không chạy trên firmware ESP32 hiện tại.

Nhận dạng cử chỉ hiện chỉ hiển thị nhãn và điểm số, chưa kích hoạt hành động nào. Kết quả phụ thuộc dữ liệu và camera; nên thử bằng ảnh/video mới trước khi sử dụng.

## 1. Trước hết: camera và IP

1. Mở `esp32s3_gesture.ino` trong Arduino IDE, chọn **ESP32S3 Dev Module**, bật PSRAM theo hướng dẫn board (N16R8 thường là OPI PSRAM), USB CDC On Boot bật nếu dùng cổng USB native, và chọn partition phù hợp.
2. Firmware mặc định tạo Wi-Fi AP `ESP32-GESTURE` (mật khẩu `12345678`) và đặt IP `192.168.4.1`. Kết nối laptop vào AP này rồi mở `http://192.168.4.1`. Không cần router; internet laptop có thể tạm ngắt khi đang nối AP.
3. Pin camera trong sketch hiện dùng mapping tham khảo của [IoTLabs Bài 10](https://iotlabs.vn/esp32-s3-devkitc-n16r8-cam-bai-10-demo-camera-web-server-stream-on-dinh-toi-uu-psram-heap): XCLK=15, SIOD=4, SIOC=5, D0..D7 = 10,9,46,3,8,18,17,16, VSYNC=6, HREF=7, PCLK=13. Bài 1 nhận diện board CAM có đầu DVP 24-pin nhưng không liệt kê map từng GPIO; Bài 10 gọi map là tham khảo, và Bài 13 trong cùng series lại đưa một mapping CAM phổ biến khác. Vì vậy mapping trong sketch chưa được xác nhận cho đúng lô board/camera M21-45 của bạn.
4. Nạp sketch trước khi thu mẫu. Serial Monitor ở 115200 sẽ in AP/IP; trang `/` xem camera, `/status` đọc JSON, `http://192.168.4.1:81/stream` là MJPEG.

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

Script xuất file `.h` và `.tflite` theo tên `--out`. Model outline dùng Sobel nhẹ để nhấn nét cạnh; firmware ưu tiên `model_outline.h` và áp dụng cùng phép biến đổi. Nếu dùng model cũ `model_data.h`, firmware giữ tiền xử lý grayscale cũ. Đặt đúng header cạnh `.ino`, mở lại/compile sketch trong Arduino IDE rồi upload. Nếu thư viện `Chirale_TensorFlowLite` thiếu, cài thư viện cùng tên trong Library Manager.

## 4. Thử model trên laptop trước khi nạp

`test_ai.py` là file chung để trích landmark từ ảnh mẫu, train classifier riêng của bạn trên các khớp đó, rồi chạy classifier cùng bộ xương tay trên webcam, stream ESP32, hoặc ảnh tĩnh. Nó vẫn có thể chạy CNN ảnh cũ để so sánh. Cài dependencies:

```powershell
py -m pip install -r requirements-collect.txt
py -m pip install -r requirements-landmark.txt
```

MediaPipe phát hiện bàn tay và cung cấp 21 khớp; TensorFlow train classifier trên 63 giá trị landmark đã chuẩn hóa. Lệnh train đọc ảnh bạn đã lấy trong `dataset`, bỏ qua ảnh không thấy bàn tay và lưu `model_landmark.keras`:

```powershell
py -m pip install -r requirements-train.txt
py test_ai.py --train --data dataset --epochs 40
```

Đảm bảo mỗi lớp có ảnh bàn tay nhìn rõ. Lớp `none` cần có các tư thế tay khác ngoài bốn cử chỉ mục tiêu; ảnh không có tay tự được trả về `none` lúc chạy. Script tự tải `hand_landmarker.task` lần đầu. Augmentation khi train phản chiếu tọa độ landmark để hỗ trợ cả hai tay; webcam không bị lật.

Webcam laptop:

```powershell
py test_ai.py --camera 0
```

Camera ESP32: nạp firmware camera trước, nối laptop vào Wi-Fi `ESP32-GESTURE`, rồi chạy:

```powershell
py test_ai.py --url http://192.168.4.1:81/stream
```

Trong cửa sổ video, nhấn `I` để mở hộp thoại chọn ảnh tĩnh; nhấn `Q` hoặc `ESC` để thoát. Hoặc chạy thẳng một ảnh:

```powershell
py test_ai.py --image "C:\duong\dan\anh.jpg"
```

Chế độ mặc định `both` vẽ bộ xương 21 khớp và hiện nhãn classifier landmark bạn vừa train, đồng thời hiển thị kết quả CNN ảnh cũ cùng ô `OUTLINE`. Cả hai dự đoán được ổn định theo cửa sổ frame để giảm nhấp nháy. Nhấn `I` để chọn ảnh tĩnh; đóng ảnh thì live tiếp tục. `--mode landmark` chỉ chạy xương tay và classifier học từ dataset của bạn; `--mode cnn` chỉ chạy CNN ảnh cũ.

Trong chế độ stream ESP32, board chỉ gửi video; landmark detector và classifier chạy trên laptop. Muốn chuyển cả pipeline lên ESP32-S3 cần port sang ESP-IDF/ESP-DL; model `.keras` này không nạp trực tiếp vào sketch Arduino hiện tại.

## 5. Kiểm tra nhận dạng trên ESP32

Mở trang IP in ở Serial Monitor. Khi có model, trang hiển thị lớp và score từng lớp; `/status` trả JSON. Ngưỡng mặc định trong sketch là `0.70`; kết quả dưới ngưỡng hiện `unknown`. Kiểm tra bằng tư thế chưa có trong dataset và nhiều điều kiện thực tế trước khi dùng.

## Giới hạn xác minh

- Môi trường hiện có Python 3.8, NumPy/Pillow/OpenCV/requests nhưng chưa có TensorFlow và chưa có `arduino-cli`; vì vậy chưa thể xác nhận build Arduino hoặc chạy thật pipeline train ở máy này. `requirements-train.txt` ghim TensorFlow 2.13.0 để tránh lỗi thiếu `tensorflow-intel==2.13.1` trên một số package index Windows.
- Thu bằng webcam laptop không cần ESP32; thu qua URL MJPEG thì cần nạp firmware stream trước.
- Pin camera đang theo IoTLabs Bài 10, nhưng ví dụ Bài 13 khác mapping. Cần sơ đồ DVP của đúng revision/lô board trước khi kết luận pin nào chính xác.
