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
('Suy luận tại thiết bị biên','Firmware ưu tiên model_raw_focus4.h và chạy CNN int8 trên ESP32-S3 bằng TensorFlow Lite Micro. Wi-Fi chỉ truyền video/trạng thái đến trang giám sát; phân loại ảnh thực hiện trên board.'),
('Mô hình và huấn luyện','CNN phân loại 5 lớp, đầu vào ảnh xám 96 × 96. Quy trình train, chia validation theo nhóm ảnh và xuất mô hình được mô tả tại mục 4.'),
('Tối ưu hóa và triển khai','Model được lượng tử hóa toàn phần int8 và đóng gói vào header C++. Chưa triển khai pruning hoặc bộ tăng tốc phần cứng.'),
('Đánh giá định lượng','Có các kết quả test offline; accuracy theo tay trái/phải chưa được đo riêng. Latency, FPS và điện năng trên board cần được đo theo bảng tại mục 5.'),
('Giao diện','Firmware cung cấp Wi-Fi AP, video MJPEG, trang web nội bộ và endpoint /status để theo dõi nhãn nhận dạng.'),
('Trạng thái prototype','Model, firmware và giao diện đã được chuẩn bị cho demo. Thử nghiệm thực tế ghi nhận tay phải dùng được trong demo; tay trái có accuracy thấp hơn. Cần lượng hóa hai kết quả bằng bộ test riêng.')], [1.55,4.85])

doc.add_heading('3. Kiến trúc và luồng xử lý',1)
bullets(['Camera cung cấp ảnh QVGA 320 × 240; firmware cắt vùng giữa, chuyển grayscale, thu nhỏ về 96 × 96 và lượng tử hóa int8. Model hiện tại dùng ảnh grayscale gốc, không áp dụng outline/Sobel.','CNN gồm ba khối convolution 3 × 3 có 16, 32 và 48 bộ lọc, max pooling, Flatten, Dense 64 ReLU, dropout 0,35 và softmax 5 lớp.','Firmware nạp model_raw_focus4.h vào TensorFlow Lite Micro và thực hiện inference trên ESP32-S3. Model nhị phân TFLite có dung lượng 468.888 byte; tensor arena được cấp phát 1 MiB trong PSRAM.','Nhãn hiển thị được làm mượt theo frame và cần confidence từ 0,55. Để phát lệnh điều khiển LED, firmware yêu cầu confidence từ 0,85 và cử chỉ ổn định 4 frame liên tiếp.','Trang web nội bộ hiển thị video, lớp nhận dạng, confidence, thời gian inference và trạng thái hai LED; /status trả dữ liệu giám sát. Wi-Fi không tham gia suy luận.'])

doc.add_heading('4. Cách thu dữ liệu và huấn luyện',1)
doc.add_heading('4.1. Nguồn và số lượng dữ liệu',2)
p('Bộ CNN hiện tại được ghép trong dataset_train_complete. Manifest.csv và SUMMARY.txt ghi nhận 7.663 ảnh sau khi loại trùng hash. Bộ dữ liệu gồm ảnh thu tại chỗ trong dataset, ảnh biến đổi từ hai cấu hình augmentation, ảnh tải từ HandGesture-Recognizer (MIT) và ảnh thuộc phần train của HaGRID. Không đưa HaGRID validation/test vào huấn luyện.')
table(['Lớp','Ảnh tại chỗ','Ảnh biến đổi','Ảnh MIT tải mạng','HaGRID train','Tổng'],[
('fist','496','600','50','445','1.591'),('palm','190','600','50','472','1.312'),('peace','491','600','50','794','1.935'),('thumbs_up','498','600','50','394','1.542'),('none','400','600','0','283','1.283'),('Tổng','2.075','3.000','200','2.388','7.663')], [1.05,1.05,1.1,1.2,1.15,.85])
p('Ảnh tại chỗ được lưu theo năm nhãn; `other` được ánh xạ sang `none`. Bộ tải mạng gồm 200 ảnh, 50 ảnh cho mỗi cử chỉ fist, palm, peace và thumbs_up; không có ảnh online cho none. HaGRID được đọc từ tinyml_improved_20261016.zip, ánh xạ open→palm và thumb_up→thumbs_up; chỉ phần poses/train được dùng. Ảnh test_hagrid gồm 75 ảnh được giữ riêng, còn test_images 25 ảnh là tập con trùng với HaGRID test nên không được tính thành nguồn train hoặc một tập đánh giá độc lập nữa.')
p('Ba nghìn ảnh biến đổi tương ứng 600 ảnh cho mỗi lớp: 100/lớp từ dataset_augmented và 500/lớp từ dataset_augmented_v2. Script sinh ảnh bằng xoay ±18°, scale 0,70–1,40, dịch chuyển tối đa ±10 pixel, thay đổi sáng/tương phản 0,70–1,30; một phần ảnh được làm mờ hoặc thêm nhiễu. Các ảnh biến đổi được giữ cùng nhóm với ảnh nguồn để tránh rò rỉ sang validation.')
doc.add_heading('4.2. Huấn luyện và xuất mô hình',2)
p('Mô hình nhận đầu vào ảnh xám 96 × 96 × 1. Kiến trúc gồm ba lớp convolution 3 × 3 lần lượt có 16, 32 và 48 bộ lọc, mỗi khối theo sau bởi max pooling; tiếp đến là Flatten, lớp Dense 64 nút với ReLU, dropout 0,35 và lớp đầu ra Dense 5 nút softmax. Mô hình được tối ưu bằng Adam với sparse categorical cross-entropy và accuracy làm chỉ số theo dõi.')
p('Lệnh train của candidate hiện tại dùng seed 43, ảnh grayscale không outline, tối đa 25 epoch, batch size 32, validation theo nhóm 20%, Adam và sparse categorical cross-entropy. Ảnh train được lật ngang trong lúc train. Class weight được cân bằng tự động, sau đó tăng hệ số lớp thumbs_up và none lên 4. EarlyStopping theo validation accuracy với patience 8; ReduceLROnPlateau theo validation loss. Converter lượng tử hóa toàn phần int8 và hiệu chuẩn bằng tối đa 40 mẫu mỗi lớp.')
for cmd in ['py prepare_cnn_data.py --hagrid-zip tinyml_improved_20261016.zip','py train_model.py --data dataset_train_complete --epochs 25 --seed 43 --no-outline --class-multiplier thumbs_up=4 --class-multiplier none=4 --out model_raw_focus4.h']:
 q=doc.add_paragraph(); q.paragraph_format.left_indent=Inches(.2); r=q.add_run(cmd); r.font.name='Consolas'; r.font.size=Pt(9)
p('Lệnh ghép dữ liệu tạo 7.663 ảnh tổng hợp; split validation theo nhóm là 20%. Khi tái lập code với seed 43 trên MANIFEST hiện tại, số tệp là 6.123 train và 1.540 validation; lật ngang tập train tạo 12.246 lượt mẫu trong quá trình fit. Validation chỉ theo dõi quá trình train và chọn checkpoint, không dùng để tính accuracy test cuối. Tệp model_raw_focus4.tflite hiện lưu có dung lượng 468.888 byte; model_raw_focus4.h nhúng mảng byte đó để firmware sử dụng.')

doc.add_heading('5. Cách kiểm tra và kết quả hiện có',1)
doc.add_heading('5.1. Kiểm tra mô hình trước khi lên board',2)
bullets(['Dành riêng validation để theo dõi quá trình train; không dùng validation thay cho đánh giá cuối.','Tạo một tập test độc lập, cân bằng giữa năm lớp và không đưa ảnh test vào train, augmentation hay tinh chỉnh.','Báo cáo accuracy tổng thể, precision/recall từng lớp và confusion matrix; chú ý peace và none vì lỗi giữa các lớp có thể bị che bởi accuracy trung bình.','Kiểm tra thêm ảnh/video từ camera thật trong nhiều điều kiện ánh sáng, khoảng cách, nền và người dùng. Ghi lại số mẫu, cấu hình, seed và phiên bản model.'])
doc.add_heading('5.2. Độ chính xác mô hình hiện có',2)
p('Candidate model_raw_focus4 đã được kiểm tra ngoại tuyến trên HaGRID test và tập người dùng. Thư mục test_images chỉ gồm 25 ảnh lấy từ chính HaGRID test nên hai kết quả này không phải hai phép thử độc lập. Tập người dùng hiện có 5 ảnh mỗi lớp; mẫu ít nên độ chính xác từng lớp thay đổi theo bước 20 điểm phần trăm.')
table(['Tập đánh giá','Số đúng / tổng','Accuracy','Ghi chú'],[
('test_images','6 / 25','24,0%','Là tập con trùng với HaGRID test.'),('HaGRID test','20 / 75','26,7%','Tập test riêng; không đưa vào train.'),('Tập người dùng','6 / 25','24,0%','Tổng hợp từ các lớp dưới đây.')], [1.35,1.25,1.05,3.0])
table(['Lớp trên tập người dùng','Đúng / mẫu','Accuracy'],[
('fist','2 / 5','40%'),('palm','2 / 5','40%'),('peace','2 / 5','40%'),('thumbs_up','0 / 5','0%'),('none','0 / 5','0%')], [2.5,1.6,2.55])
p('Trong thử nghiệm trực tiếp hiện tại, người dùng ghi nhận tay phải được nhận dạng đủ để sử dụng trong demo, còn accuracy tay trái thấp hơn và chưa ổn định. Chưa có số lượng mẫu tách riêng theo tay trái/phải nên không quy đổi nhận xét này thành phần trăm. Kết quả offline trên tập người dùng cho thấy thumbs_up và none còn yếu; cần xem nhận dạng tay phải là khả năng demo theo điều kiện đã thử, không phải độ chính xác đã được xác nhận trên toàn bộ người dùng/cử chỉ.')
table(['Tay','Hiện trạng thử trực tiếp','Số đo định lượng'],[
('Tay phải','Nhận dạng đủ dùng cho demo theo quan sát hiện tại.','Chưa tách mẫu và chưa tính accuracy riêng.'),('Tay trái','Accuracy thấp hơn tay phải; nhận dạng chưa ổn định.','Chưa tách mẫu và chưa tính accuracy riêng.')], [1.0,3.0,2.65])
p('Có thể tái lập kiểm tra ảnh bằng `py test_ai.py --test-dir test_images --model model_raw_focus4.tflite`. Khi so sánh hoặc báo cáo accuracy mới, không gộp test_images và HaGRID test thành một mẫu độc lập vì có ảnh trùng nhau.')
table(['Chỉ số hiệu năng','Kết quả hiện có','Phạm vi đo / bước tiếp theo'],[
('Thời gian suy luận CNN','Chưa có giá trị số được lưu. Firmware ghi ms trong /status và Serial Monitor.','Trường hiện tại đo riêng Invoke() với độ phân giải 1 ms, không gồm lấy ảnh/tiền xử lý. Đo ≥100 lần sau warm-up; báo cáo trung bình và P95.'),
('FPS / throughput','Chưa có phép đo lưu lại.','Đếm frame được phân loại hoàn tất trong 30–60 giây; ghi cấu hình camera và có/không tính thời gian truyền web.'),
('Bộ nhớ','Tensor arena được cấp phát 1 MiB trong PSRAM; model nhị phân 468.888 byte.','Ghi arena_used_bytes và heap/PSRAM còn trống sau khởi tạo, trong lúc chạy.'),
('Điện năng','Chưa đo.','Đo dòng/áp nguồn khi idle và nhận diện; tính P=V×I, có thể báo cáo Wh/giờ hoặc năng lượng mỗi lần suy luận.')], [1.25,2.25,2.9])
doc.add_heading('5.3. Đánh giá tối ưu hóa theo tiêu chí cuộc thi',2)
p('Bản demo áp dụng lượng tử hóa toàn phần int8; model_raw_focus4.tflite có dung lượng 468.888 byte. Chưa có đối chứng float/int8, pruning hoặc bộ tăng tốc phần cứng, vì vậy chưa thể lượng hóa mức tăng tốc hay điện năng tiết kiệm. Kết quả test hiện tại phản ánh độ chính xác của candidate; các số đo latency/FPS/nguồn điện cần thu trên ESP32-S3 trước khi kết luận hiệu quả tối ưu hóa.')
doc.add_heading('5.4. Phương pháp triển khai và nhận diện trên ESP32-S3',2)
p('Đặt model_raw_focus4.h cạnh esp32s3_gesture.ino. Trong Arduino IDE chọn ESP32S3 Dev Module, bật PSRAM (OPI cho N16R8), chọn partition phù hợp và cài Chirale_TensorFlowLite; biên dịch rồi nạp sketch. Khi khởi động, firmware cấp phát tensor arena 1 MiB ở PSRAM, kiểm tra input int8 1 × 96 × 96 × 1 và khởi tạo TensorFlow Lite Micro. Camera tạo frame 320 × 240; board cắt hình vuông ở giữa, chuyển grayscale, resize 96 × 96 và lượng tử hóa theo scale/zero-point trước khi gọi Invoke(). Mô hình không dùng outline/Sobel ở cấu hình này.')
p('Board phát Wi-Fi AP ESP32-GESTURE (mật khẩu 12345678) tại 192.168.4.1. Trang web hiển thị video, nhãn, score và trạng thái LED; /status trả confidence, các score lớp, thời gian inference và LED. Phân loại chạy trên ESP32-S3; trang web không thực hiện suy luận. Pin camera phải khớp revision kit khi thử thực tế.')
doc.add_heading('5.5. Số frame cần để nhận diện',2)
p('Nhận diện hiển thị và phát lệnh LED dùng hai điều kiện khác nhau. Label hiển thị khi confidence sau làm mượt đạt 0,55. Bộ làm mượt là sₙ = 0,5sₙ₋₁ + 0,5p, với p là xác suất hiện tại. Nếu p ổn định và trạng thái bắt đầu từ 0, có thể ước lượng số frame như sau:')
table(['p ổn định qua các frame','Frame để hiện nhãn (ngưỡng 0,55)','Frame để lệnh LED (ngưỡng 0,85 và 4 frame ổn định)'],[
('1,00','2','6'),('0,95','2','7'),('0,90','2','8'),('0,70','3','Không phát lệnh'),('≤0,55','Không vượt ngưỡng','Không phát lệnh')], [1.65,2.15,2.65])
p('Số frame lệnh LED gồm thời gian confidence sau làm mượt vượt 0,85 và đủ 4 lần inference liên tiếp cùng cử chỉ. Đây là ước tính khi xác suất gần như ổn định và bộ làm mượt bắt đầu từ 0; trong sử dụng thực tế confidence thay đổi từng frame và giữ trạng thái từ frame trước, nên thời gian có thể khác. Nếu confidence không đạt 0,85, giao diện vẫn có thể hiện nhãn nhưng firmware không phát lệnh LED.')

doc.add_heading('6. Giới hạn và việc cần hoàn thành',1)
table(['Ưu tiên','Việc cần làm để chốt demo'],[
('1','Xác nhận camera và model_raw_focus4 chạy đúng trên kit; camera pin phải khớp revision board.'),
('2','Bổ sung ảnh tay trái từ nhiều người, góc, khoảng cách và ánh sáng; cân nhắc cân bằng hai bên tay trong dữ liệu train.'),
('3','Thu thêm mẫu thumbs_up và none từ đúng camera để xử lý các lớp đang có 0/5 ở tập người dùng.'),
('4','Đo lại accuracy trên tập test độc lập; tính riêng theo tay trái/phải, từng lớp, latency, FPS, bộ nhớ và điện năng.'),
('5','Ghi log phiên bản model, cấu hình threshold, điều kiện test và minh chứng cho demo web/LED.')], [0.7,5.7])
p('Kết quả giữa tay trái/phải có thể chịu ảnh hưởng của hướng bàn tay, vị trí trong khung và cách thu mẫu. So sánh công bằng cần cùng số lượng mẫu, cùng nhóm cử chỉ và cùng điều kiện ghi hình cho mỗi bên tay.')

doc.add_heading('7. Kết luận',1)
p('Prototype hiện có bộ dữ liệu tổng hợp 7.663 ảnh, CNN lượng tử hóa int8, firmware suy luận trên ESP32-S3 và giao diện web giám sát. Theo kiểm tra sử dụng hiện tại, tay phải đã đủ dùng cho demo nhưng tay trái nhận dạng yếu hơn. Các tập đánh giá offline còn cho accuracy thấp, đặc biệt ở thumbs_up và none; do đó cần tiếp tục cân bằng dữ liệu hai bên tay, cải thiện các lớp yếu và đo latency/FPS/điện năng trên kit trước khi xem đây là kết quả định lượng hoàn chỉnh.')

doc.add_heading('Nguồn nội bộ đã đối chiếu',1)
p('Kế hoạch và thể lệ Edge AI Design Challenge, số 38/KH-ĐĐT ngày 16/09/2026; README dự án; SUMMARY.txt và MANIFEST.csv trong dataset_train_complete; prepare_cnn_data.py; train_model.py; test_ai.py; esp32s3_gesture.ino. Số mẫu theo nguồn được tổng hợp từ manifest của bộ dữ liệu huấn luyện hiện tại.')

for sec in doc.sections:
 f=sec.footer.paragraphs[0]; f.alignment=WD_ALIGN_PARAGRAPH.CENTER; f.add_run('Báo cáo demo Edge AI • ESP32-S3').font.size=Pt(8)
doc.core_properties.title='Báo cáo demo Edge AI nhận diện cử chỉ tay trên ESP32-S3'
doc.core_properties.subject='Quy trình huấn luyện và kiểm tra prototype'
doc.save(out)
print(out)
