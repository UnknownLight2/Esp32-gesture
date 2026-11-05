from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

out='reports/Bao_cao_demo_EdgeAI_nhan_dien_cu_chi_ESP32S3.docx'
Path(out).parent.mkdir(parents=True, exist_ok=True)
doc=Document()
sec=doc.sections[0]
sec.top_margin=Inches(.7); sec.bottom_margin=Inches(.7); sec.left_margin=Inches(.8); sec.right_margin=Inches(.8)
styles=doc.styles
styles['Normal'].font.name='Aptos'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'),'Aptos'); styles['Normal'].font.size=Pt(10.5)
styles['Normal'].paragraph_format.space_after=Pt(6)
for nm,size in [('Title',25),('Heading 1',16),('Heading 2',12)]:
 s=styles[nm]; s.font.name='Aptos Display'; s._element.rPr.rFonts.set(qn('w:eastAsia'),'Aptos Display'); s.font.size=Pt(size); s.font.color.rgb=RGBColor(0,0,0); s.font.bold=True
styles['Heading 1'].paragraph_format.space_before=Pt(13); styles['Heading 1'].paragraph_format.space_after=Pt(5)
styles['Heading 2'].paragraph_format.space_before=Pt(8); styles['Heading 2'].paragraph_format.space_after=Pt(3)

def p(text='',style=None,boldlead=None):
 q=doc.add_paragraph(style=style)
 if boldlead and text.startswith(boldlead): q.add_run(boldlead).bold=True; q.add_run(text[len(boldlead):])
 else: q.add_run(text)
 return q
def bullets(items):
 for x in items: doc.add_paragraph(x,style='List Bullet')
def table(headers,rows,widths=None):
 t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
 for i,h in enumerate(headers):
  c=t.rows[0].cells[i]; c.text=h; c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
  sh=OxmlElement('w:shd'); sh.set(qn('w:fill'),'17365D'); c._tc.get_or_add_tcPr().append(sh)
  for r in c.paragraphs[0].runs: r.font.bold=True; r.font.color.rgb=RGBColor(255,255,255); r.font.size=Pt(9)
 for row in rows:
  cells=t.add_row().cells
  for i,v in enumerate(row): cells[i].text=str(v); cells[i].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
 for row in t.rows:
  for c in row.cells:
   for pp in c.paragraphs:
    pp.paragraph_format.space_after=Pt(2)
    for rr in pp.runs: rr.font.size=Pt(9)
 if widths:
  for row in t.rows:
   for i,w in enumerate(widths): row.cells[i].width=Inches(w)
 doc.add_paragraph()
 return t

title=doc.add_paragraph(style='Title'); title.add_run('BÁO CÁO DEMO EDGE AI\nNHẬN DIỆN CỬ CHỈ TAY TRÊN ESP32-S3')
sub=p('Prototype học thuật | Trạng thái phát triển khoảng 80–90% theo phạm vi demo')
sub.alignment=WD_ALIGN_PARAGRAPH.LEFT
p('Báo cáo mô tả mô hình nhận diện cử chỉ tay, quy trình huấn luyện và tối ưu hóa, phương án suy luận trên ESP32-S3, giao diện giám sát cùng kế hoạch đánh giá định lượng. Hệ thống được thiết kế để CNN chạy trực tiếp trên vi điều khiển; việc xác nhận đầy đủ trên kit thực và đo các chỉ số hiệu năng là bước kiểm tra cần hoàn tất cho bản demo.')

doc.add_heading('1. Tóm tắt đề tài',1)
p('Đề tài xây dựng prototype nhận diện năm trạng thái bàn tay: fist, palm, peace, thumbs_up và none. Ảnh được tiền xử lý thành ảnh xám 96 × 96, phân loại bằng mạng CNN nhỏ và lượng tử hóa int8 để tích hợp với TensorFlow Lite Micro. ESP32-S3 cung cấp luồng hình ảnh qua Wi-Fi và trang web hiển thị kết quả nhận diện. Prototype đã có các thành phần chính; bước hoàn thiện hiện nay là kiểm tra trực tiếp trên kit, đo hiệu năng và bổ sung dữ liệu từ camera sử dụng trong demo.')

doc.add_heading('2. Yêu cầu kỹ thuật và trạng thái triển khai',1)
table(['Yêu cầu','Thiết kế hiện tại và phần cần xác nhận'],[
('Suy luận tại thiết bị biên','Mục tiêu là xử lý ảnh và chạy CNN trên ESP32-S3 bằng TensorFlow Lite Micro. Wi-Fi dùng để truy cập giao diện và luồng camera; không gửi ảnh lên cloud để phân loại. Cần kiểm tra inference hoạt động trên đúng kit sau khi nạp.'),
('Mô hình và huấn luyện','CNN phân loại 5 lớp, đầu vào ảnh xám 96 × 96. Quy trình train, chia validation theo nhóm ảnh và xuất mô hình được mô tả tại mục 4.'),
('Tối ưu hóa và triển khai','Đã có bước lượng tử hóa int8 và xuất header C++ cho firmware. Chưa triển khai pruning hoặc bộ tăng tốc phần cứng; mức cải thiện tốc độ/tài nguyên cần đo trên board.'),
('Đánh giá định lượng','Cần báo cáo accuracy tổng thể và từng lớp, latency, FPS, bộ nhớ và điện năng trên ESP32-S3; cách đo được nêu tại mục 5.'),
('Giao diện','Firmware cung cấp Wi-Fi AP, video MJPEG, trang web nội bộ và endpoint /status để theo dõi nhãn nhận dạng.'),
('Trạng thái prototype','Các thành phần chính đã có trong dự án. Việc hoàn tất demo cần xác nhận camera, inference trên kit và thu số liệu đo thực nghiệm.')], [1.55,4.85])

doc.add_heading('3. Kiến trúc và luồng xử lý',1)
bullets(['Camera cung cấp ảnh đầu vào; firmware tiền xử lý ảnh và thực hiện inference CNN trên ESP32-S3.','Tiền xử lý ảnh: chuyển grayscale, cắt vuông chính giữa, resize 96 × 96 và áp dụng bộ lọc outline/Sobel thống nhất giữa huấn luyện và firmware.','CNN gồm ba khối convolution (16, 32, 48 kênh), pooling, fully connected 64 nút, dropout và softmax 5 lớp.','Mô hình được chuyển sang TFLite, lượng tử hóa int8 bằng representative samples lấy từ đủ các lớp, sau đó đóng gói thành header C++ để firmware nạp vào TensorFlow Lite Micro.','Firmware làm mượt xác suất qua các frame và hiển thị unknown nếu điểm cao nhất dưới ngưỡng 0,70.','Trang web nội bộ hiển thị video và nhãn; endpoint /status cung cấp trạng thái để giám sát. Wi-Fi phục vụ truyền video/trạng thái, còn phân loại được thiết kế chạy trên board.'])

doc.add_heading('4. Cách thu dữ liệu và huấn luyện',1)
doc.add_heading('4.1. Chuẩn bị dữ liệu',2)
p('Tạo thư mục dataset gồm fist, palm, peace, thumbs_up và none (có thể dùng tên other cho lớp none). Thu vài trăm ảnh/lớp làm mục tiêu khởi đầu; thay đổi người, tay trái/phải, góc xoay, khoảng cách, nền và ánh sáng. Lớp none nên gồm tư thế tay ngoài bốn cử chỉ và khung không có tay. Tránh chỉ chụp một chuỗi frame gần như giống nhau.')
p('Có thể thu bằng webcam laptop để phát triển sớm, sau đó bổ sung ảnh từ chính camera ESP32-S3 vì khác biệt về góc nhìn, màu và tiêu cự có thể làm giảm kết quả khi chuyển thiết bị. Giữ các ảnh liên quan/augmentation và các frame chụp sát nhau trong cùng nhóm khi chia train-validation để hạn chế rò rỉ dữ liệu.')
doc.add_heading('4.2. Huấn luyện và xuất mô hình',2)
p('Mô hình nhận đầu vào ảnh xám 96 × 96 × 1. Kiến trúc gồm ba lớp convolution 3 × 3 lần lượt có 16, 32 và 48 bộ lọc, mỗi khối theo sau bởi max pooling; tiếp đến là Flatten, lớp Dense 64 nút với ReLU, dropout 0,35 và lớp đầu ra Dense 5 nút softmax. Mô hình được tối ưu bằng Adam với sparse categorical cross-entropy và accuracy làm chỉ số theo dõi.')
p('Quy trình dự án tạo dữ liệu biến đổi bằng augment_dataset.py rồi huấn luyện bằng train_model.py. Các phép biến đổi gồm xoay ngẫu nhiên ±18°, thay đổi tỉ lệ 0,70–1,40, dịch chuyển tối đa ±10 pixel, thay đổi độ sáng/độ tương phản 0,70–1,30; một phần ảnh được làm mờ nhẹ hoặc thêm nhiễu. Ảnh gốc và ảnh biến đổi được giữ cùng nhóm khi chia train-validation. Khi train, ảnh trong tập train được lật ngang thêm một lần để mô hình học cả hai hướng tay.')
p('Tách validation mặc định là 20% theo nhóm ảnh; seed mặc định là 7. Huấn luyện tối đa 25 epoch, batch size 32, class weight cân bằng các lớp; EarlyStopping theo validation accuracy với patience 8 và ReduceLROnPlateau theo validation loss. Sau đó TFLite Converter áp dụng tối ưu hóa và lượng tử hóa int8 toàn phần, lấy tối đa 40 ảnh đại diện mỗi lớp để hiệu chuẩn. Mô hình int8 đầu vào/đầu ra được đóng gói thành .tflite và mảng C++ trong header.')
for cmd in ['py augment_dataset.py --data dataset --out dataset_augmented --per-class 100','py train_model.py --data dataset_augmented --epochs 25 --out model_outline.h']:
 q=doc.add_paragraph(); q.paragraph_format.left_indent=Inches(.2); r=q.add_run(cmd); r.font.name='Consolas'; r.font.size=Pt(9)
p('Firmware hiện ưu tiên model_outline.h. README ghi lệnh tạo model này từ dataset_augmented; tệp model_outline.tflite có ngày cập nhật 23/10/2026. Bộ dataset_augmented lưu trong project có 1.161 ảnh gốc và 500 ảnh biến đổi (1.661 tệp trước khi chia). Các ảnh gốc trong bộ này thuộc snapshot trước đợt thu mẫu mới ngày 02/11/2026.')
table(['Nhãn','Ảnh thu tại chỗ trong snapshot','Ảnh biến đổi','Tổng trong bộ model'],[
('fist','190','100','290'),('palm','190','100','290'),('peace','190','100','290'),('thumbs_up','191','100','291'),('none (other)','400','100','500'),('Tổng','1.161','500','1.661')], [1.35,2.15,1.35,1.55])
p('Theo thuật toán chia nhóm hiện tại với seed mặc định 7, có thể tái lập split 1.353 tệp train và 308 validation; lật ngang tập train làm thành 2.706 lượt mẫu đưa vào fit. Repository không lưu log của lần train model_outline để xác nhận chính xác split lịch sử hoặc accuracy tại thời điểm đó, nên các con số split này được ghi là tái lập từ dữ liệu và mã nguồn đang có.')
doc.add_heading('4.3. Dữ liệu mới trong project chưa nằm trong model_outline hiện tại',2)
table(['Nhãn','Ảnh hiện có trong dataset','Ảnh trong snapshot model_outline','Bổ sung sau snapshot'],[
('fist','496','190','306'),('palm','190','190','0'),('peace','491','190','301'),('thumbs_up','498','191','307'),('none (other)','400','400','0'),('Tổng ảnh tại chỗ','2.075','1.161','914')], [1.35,1.75,1.8,1.5])
p('Các tệp mới làm tổng ảnh thu tại chỗ trong dataset tăng lên 2.075. Chúng chưa được đưa vào model_outline.tflite hiện có: thư mục dataset_augmented vẫn chứa snapshot cũ. Để dùng các ảnh này cho CNN, cần tạo lại dataset_augmented từ dataset hiện tại, train lại, xuất model_outline.h/.tflite và đánh giá lại trên tập test độc lập.')
p('Ngoài ra có 200 ảnh tải từ kho HandGesture-Recognizer, gồm 50 ảnh mỗi lớp fist, palm, peace và thumbs_up. Hồ sơ dự án ghi bộ ảnh này dùng trong một thử nghiệm phân loại khác, không thuộc quy trình CNN hiện tại, nên không cộng vào số mẫu model_outline. dataset_augmented_v2 có 2.500 ảnh biến đổi trên snapshot cũ (500 mỗi lớp), tổng bộ 3.661 tệp; README không chỉ định bộ này làm nguồn tạo model_outline nên cũng không cộng vào số mẫu của model đang được firmware ưu tiên.')
p('Tệp model_outline.tflite hiện lưu có dung lượng 468.888 byte. Header C++ lớn hơn do biểu diễn từng byte dưới dạng văn bản; kích thước dữ liệu mô hình nhị phân khi chạy vẫn là 468.888 byte.')

doc.add_heading('5. Cách kiểm tra và kết quả hiện có',1)
doc.add_heading('5.1. Kiểm tra mô hình trước khi lên board',2)
bullets(['Dành riêng validation để theo dõi quá trình train; không dùng validation thay cho đánh giá cuối.','Tạo một tập test độc lập, cân bằng giữa năm lớp và không đưa ảnh test vào train, augmentation hay tinh chỉnh.','Báo cáo accuracy tổng thể, precision/recall từng lớp và confusion matrix; chú ý peace và none vì lỗi giữa các lớp có thể bị che bởi accuracy trung bình.','Kiểm tra thêm ảnh/video từ camera thật trong nhiều điều kiện ánh sáng, khoảng cách, nền và người dùng. Ghi lại số mẫu, cấu hình, seed và phiên bản model.'])
doc.add_heading('5.2. Kết quả và cách ghi nhận',2)
p('Trong dự án có thư mục test_images gồm 25 ảnh, mỗi lớp 5 ảnh. Tuy nhiên chưa có kết quả chạy model_outline.tflite trên tập này được lưu; 5 mẫu/lớp cũng chỉ phù hợp kiểm tra nhanh vì mỗi lỗi làm thay đổi accuracy lớp 20 điểm phần trăm. Cần giữ một tập test độc lập lớn hơn, không dùng trong train/augmentation, rồi điền kết quả đo thực tế. Không dùng accuracy validation thay cho test cuối.')
table(['Lớp','Số ảnh test hiện có','Đúng','Accuracy hiện ghi nhận'],[
('fist','5','Chưa đo','Chưa đo'),('palm','5','Chưa đo','Chưa đo'),('peace','5','Chưa đo','Chưa đo'),('thumbs_up','5','Chưa đo','Chưa đo'),('none','5','Chưa đo','Chưa đo'),('Tổng','25','Chưa đo','Chưa đo')], [1.15,1.35,1.1,2.8])
p('Có thể đánh giá ảnh tĩnh bằng `py test_ai.py --test-dir test_images --model model_outline.tflite --outline`. Script trả nhãn dự đoán, accuracy và recall theo lớp khi thư mục con được đặt đúng tên. Sau đó nên thử thêm video camera để đánh giá ánh sáng, rung, khoảng cách và độ trễ toàn hệ thống.')
table(['Chỉ số hiệu năng','Giá trị/ghi nhận hiện tại','Cách đo trên kit'],[
('Thời gian suy luận CNN','Chưa có số đo thực nghiệm được lưu. Firmware ghi trường ms trong /status và Serial Monitor.','Lấy ít nhất 100 lần đo sau warm-up; báo cáo trung bình, min/max và P95. Bộ đếm hiện tại đo riêng thời gian Invoke, độ phân giải 1 ms.'),
('FPS nhận diện','Chưa đo.','Đếm số kết quả inference hoàn tất trong 30–60 giây; báo cáo cùng điều kiện camera và Wi-Fi.'),
('Bộ nhớ','Tensor arena được cấp phát 1 MiB trong PSRAM; firmware in số byte arena thực dùng khi khởi tạo model.','Ghi arena_used_bytes, heap/PSRAM còn trống trước và sau khởi tạo, trong lúc chạy.'),
('Điện năng','Chưa đo.','Đo dòng và điện áp tại nguồn board khi idle và nhận diện; tính công suất P=V×I, nếu có thể ghi năng lượng trung bình mỗi giây và mỗi lần inference.')], [1.25,2.1,3.05])
doc.add_heading('5.3. Đánh giá tối ưu hóa theo tiêu chí cuộc thi',2)
p('Bản demo hiện áp dụng lượng tử hóa int8; tệp model_outline.tflite có dung lượng 468.888 byte. Chưa có số liệu so sánh float và int8 hoặc pruning/tăng tốc phần cứng trong dự án. Khi đánh giá, chạy cùng tập test và cùng điều kiện board, đối chiếu accuracy, kích thước mô hình, RAM/PSRAM, latency, FPS và điện năng để mô tả mức đánh đổi giữa độ chính xác và tài nguyên.')
doc.add_heading('5.4. Phương pháp triển khai và nhận diện trên ESP32-S3',2)
p('Sau khi huấn luyện, đặt model_outline.h cạnh sketch esp32s3_gesture.ino để firmware nạp model int8 đã nhúng. Chọn ESP32S3 Dev Module trong Arduino IDE, bật PSRAM theo loại board, chọn partition phù hợp và cài thư viện Chirale_TensorFlowLite. Khi khởi động, firmware cấp phát tensor arena 1 MiB trong PSRAM, kiểm tra kích thước input 1 × 96 × 96 × 1 và kiểu int8, rồi khởi tạo TensorFlow Lite Micro. Camera cung cấp frame 320 × 240; firmware cắt vùng giữa thành hình vuông, lấy grayscale, thu nhỏ về 96 × 96, áp dụng outline/Sobel và lượng tử hóa theo scale/zero-point của model trước khi gọi Invoke().')
p('Board tạo Wi-Fi AP ESP32-GESTURE (mật khẩu cấu hình 12345678) tại 192.168.4.1. Người dùng mở trang web để xem video và kết quả; /status trả nhãn, confidence và thời gian inference. Ảnh được phân loại bởi model trên ESP32-S3; trang web chỉ hiển thị video/trạng thái. Trước khi demo cần xác nhận pin camera phù hợp đúng revision của board.')
doc.add_heading('5.5. Số frame cần để nhận diện',2)
p('Firmware không chờ một cửa sổ cố định gồm N frame. Mỗi lần inference cập nhật trung bình mũ cho confidence: sₙ = 0,5sₙ₋₁ + 0,5p, trong đó p là xác suất frame hiện tại và s₀ ban đầu bằng 0. Kết quả được phát khi confidence đã làm mượt đạt ngưỡng 0,70. Nếu các frame liên tiếp cho cùng một xác suất p ổn định, số frame tối thiểu theo công thức là:')
table(['Confidence mỗi frame p','Số frame tích lũy để đạt 0,70','Ghi chú'],[
('1,00','2','Sau 2 frame, s=0,75.'),('0,90','3','Sau 3 frame, s=0,7875.'),('0,80','3','Sau 3 frame, s=0,70.'),('0,75','4','Sau 4 frame, s≈0,703.'),('0,70 hoặc thấp hơn','Không đạt hữu hạn nếu p ổn định','Ngưỡng firmware cần được vượt qua; kết quả thực tế thay đổi theo từng frame.')], [1.55,1.8,3.05])
p('Vì vậy không thể cam kết “nhận diện sau đúng N frame” cho mọi trường hợp. Với confidence cao, quy tắc làm mượt hiện tại có thể vượt ngưỡng sau khoảng 2–3 frame; cử chỉ mờ, che khuất hoặc confidence thấp có thể cần lâu hơn hoặc hiện unknown. Bảng trên giả sử trạng thái làm mượt bắt đầu từ 0; khi đổi cử chỉ, confidence còn lưu từ các frame trước nên số frame thực tế có thể khác. Thời gian phản hồi còn phụ thuộc FPS camera và thời gian xử lý từng frame, cần đo trên kit.')

doc.add_heading('6. Giới hạn và việc cần hoàn thành',1)
table(['Ưu tiên','Việc cần làm để chốt demo'],[
('1','Xác nhận pin camera đúng revision kit và đưa camera ESP32-S3 vào luồng chạy ổn định.'),
('2','Nạp đúng model CNN int8; kiểm tra inference thực sự trên board, không chỉ stream ảnh sang laptop.'),
('3','Thu thêm dữ liệu từ camera/kit, đặc biệt các mẫu peace và none có tư thế tay lạ; giữ test độc lập.'),
('4','Đo accuracy trên tập test mới, latency, FPS, bộ nhớ và độ ổn định; lưu cấu hình và minh chứng.'),
('5','Hoàn thiện demo web, sơ đồ hệ thống, poster và phần trình bày theo tiêu chí cuộc thi.')], [0.7,5.7])
p('Kết quả có thể thay đổi khi chuyển từ webcam sang camera board do khác biệt về góc nhìn, màu sắc và tiêu cự. Cần kiểm tra pin camera theo đúng revision phần cứng, đồng thời đánh giá riêng từng lớp trên tập test đủ đại diện.')

doc.add_heading('7. Kết luận',1)
p('Prototype đã có các thành phần chính gồm bài toán nhận dạng cử chỉ, CNN gọn, quy trình xuất mô hình int8, firmware camera và giao diện web nội bộ. Công việc tiếp theo là xác nhận inference trên ESP32-S3, đo định lượng hiệu năng và hoàn thiện kiểm tra với ảnh từ camera thực tế. Demo hiện được xem là đang hoàn thiện; số liệu phần cứng sẽ được bổ sung sau khi có kết quả đo.')

doc.add_heading('Nguồn nội bộ đã đối chiếu',1)
p('Kế hoạch và thể lệ Edge AI Design Challenge, số 38/KH-ĐĐT ngày 16/09/2026; README dự án; train_model.py; esp32s3_gesture.ino. Yêu cầu cuộc thi và tình trạng triển khai của prototype được trình bày ở các mục tương ứng.')

for sec in doc.sections:
 f=sec.footer.paragraphs[0]; f.alignment=WD_ALIGN_PARAGRAPH.CENTER; f.add_run('Báo cáo demo Edge AI • ESP32-S3').font.size=Pt(8)
doc.core_properties.title='Báo cáo demo Edge AI nhận diện cử chỉ tay trên ESP32-S3'
doc.core_properties.subject='Quy trình huấn luyện và kiểm tra prototype'
doc.save(out)
print(out)
