# ESP32-S3 gesture demo

Pipeline tạo 5 nhãn `fist`, `palm`, `peace`, `thumbs_up`, `none`. Lớp `none` là các tư thế bàn tay không thuộc bốn cử chỉ mục tiêu. Chưa gán hành động nào cho cử chỉ.

## 1. Trước hết: camera và IP

1. Mở `esp32s3_gesture.ino` trong Arduino IDE, chọn **ESP32S3 Dev Module**, bật PSRAM theo hướng dẫn board (N16R8 thường là OPI PSRAM), USB CDC On Boot bật nếu dùng cổng USB native, và chọn partition phù hợp.
2. Firmware mặc định tạo Wi-Fi AP `ESP32-GESTURE` (mật khẩu `12345678`) và đặt IP `192.168.4.1`. Kết nối laptop vào AP này rồi mở `http://192.168.4.1`. Không cần router; internet laptop có thể tạm ngắt khi đang nối AP.
3. Pin camera trong sketch hiện dùng mapping tham khảo của [IoTLabs Bài 10](https://iotlabs.vn/esp32-s3-devkitc-n16r8-cam-bai-10-demo-camera-web-server-stream-on-dinh-toi-uu-psram-heap): XCLK=15, SIOD=4, SIOC=5, D0..D7 = 10,9,46,3,8,18,17,16, VSYNC=6, HREF=7, PCLK=13. Bài 1 nhận diện board CAM có đầu DVP 24-pin nhưng không liệt kê map từng GPIO; Bài 10 gọi map là tham khảo, và Bài 13 trong cùng series lại đưa một mapping CAM phổ biến khác. Vì vậy mapping trong sketch chưa được xác nhận cho đúng lô board/camera M21-45 của bạn.
4. Nạp sketch trước khi thu mẫu. Serial Monitor ở 115200 sẽ in AP/IP; trang `/` xem camera, `/status` đọc JSON, `http://192.168.4.1:81/stream` là MJPEG.

## 2. Thu sample

Cài Python 3.8-3.11 và dependency (TensorFlow 2.13.1 có wheel cho CPython 3.8 trên Windows):

```powershell
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

```powershell
py -m pip install -r requirements-train.txt
py train_model.py --data dataset --epochs 25 --out model_data.h
```

Script xuất `model_data.h` và `model_data.tflite`. Header có thứ tự nhãn cố định như trên. Đặt `model_data.h` cạnh `.ino`, mở lại/compile sketch trong Arduino IDE rồi upload. Nếu thư viện `Chirale_TensorFlowLite` thiếu, cài thư viện cùng tên trong Library Manager.

## 4. Kiểm tra nhận dạng

Mở trang IP in ở Serial Monitor. Khi có model, trang hiển thị lớp và score từng lớp; `/status` trả JSON. Ngưỡng mặc định trong sketch là `0.70`; kết quả dưới ngưỡng hiện `unknown`. Kiểm tra bằng tư thế chưa có trong dataset và nhiều điều kiện thực tế trước khi dùng.

## Giới hạn xác minh

- Môi trường hiện có Python 3.8, NumPy/Pillow/OpenCV/requests nhưng chưa có TensorFlow và chưa có `arduino-cli`; vì vậy chưa thể xác nhận build Arduino hoặc chạy thật pipeline train ở máy này. `requirements-train.txt` ghim TensorFlow 2.13.1 để tương thích CPython 3.8.
- Thu bằng webcam laptop không cần ESP32; thu qua URL MJPEG thì cần nạp firmware stream trước.
- Pin camera đang theo IoTLabs Bài 10, nhưng ví dụ Bài 13 khác mapping. Cần sơ đồ DVP của đúng revision/lô board trước khi kết luận pin nào chính xác.
