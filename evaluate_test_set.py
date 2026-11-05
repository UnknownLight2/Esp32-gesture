#!/usr/bin/env python3
"""Evaluate the ESP32 TFLite model on labeled folders and create an editable DOCX report."""
import argparse
import csv
from datetime import date
from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import tensorflow as tf
from gesture_preprocess import apply_outline

LABELS = ("fist", "palm", "peace", "thumbs_up", "none")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE = 96
NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ET.register_namespace("w", NS_W)
ET.register_namespace("r", NS_R)


def q(tag):
    return "{%s}%s" % (NS_W, tag)


def preprocess(path, no_outline=False):
    with Image.open(path) as source:
        image = source.convert("L")
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        resampling = getattr(Image, "Resampling", Image)
        image = image.crop((left, top, left + side, top + side)).resize(
            (SIZE, SIZE), resampling.BOX
        )
        pixels = np.asarray(image, dtype=np.uint8)
        if not no_outline:
            pixels = apply_outline(pixels)
        return pixels.astype(np.float32)[None, ..., None] / 255.0


def word_run(paragraph, text, bold=False, size=22, color=None):
    run = ET.SubElement(paragraph, q("r"))
    props = ET.SubElement(run, q("rPr"))
    ET.SubElement(props, q("rFonts"), {q("ascii"): "Arial", q("hAnsi"): "Arial", q("cs"): "Arial"})
    ET.SubElement(props, q("sz"), {q("val"): str(size)})
    if bold:
        ET.SubElement(props, q("b"))
    if color:
        ET.SubElement(props, q("color"), {q("val"): color})
    node = ET.SubElement(run, q("t"))
    node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    node.text = text


def add_paragraph(body, text="", bold=False, size=22, color=None, before=0, after=100):
    p = ET.SubElement(body, q("p"))
    ppr = ET.SubElement(p, q("pPr"))
    spacing = ET.SubElement(ppr, q("spacing"), {q("before"): str(before), q("after"): str(after)})
    if text:
        word_run(p, text, bold, size, color)
    return p


def add_table(body, rows, widths=None):
    table = ET.SubElement(body, q("tbl"))
    props = ET.SubElement(table, q("tblPr"))
    borders = ET.SubElement(props, q("tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        ET.SubElement(borders, q(edge), {q("val"): "single", q("sz"): "5", q("color"): "A6A6A6"})
    for row_index, values in enumerate(rows):
        tr = ET.SubElement(table, q("tr"))
        for col_index, value in enumerate(values):
            tc = ET.SubElement(tr, q("tc"))
            tcpr = ET.SubElement(tc, q("tcPr"))
            if widths:
                ET.SubElement(tcpr, q("tcW"), {q("w"): str(widths[col_index]), q("type"): "dxa"})
            if row_index == 0:
                shade = ET.SubElement(tcpr, q("shd"), {q("fill"): "17365D"})
            p = ET.SubElement(tc, q("p"))
            ppr = ET.SubElement(p, q("pPr"))
            ET.SubElement(ppr, q("spacing"), {q("after"): "40"})
            word_run(p, str(value), bold=(row_index == 0), size=18,
                     color="FFFFFF" if row_index == 0 else None)
    add_paragraph(body, after=80)


def write_docx(path, args, results, per_class, matrix, total, correct):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    document = ET.Element(q("document"))
    body = ET.SubElement(document, q("body"))
    add_paragraph(body, "BÁO CÁO KIỂM THỬ DEMO", bold=True, size=34, color="17365D", after=80)
    add_paragraph(body, "Nhận diện cử chỉ tay bằng ESP32-S3", bold=True, size=28, after=220)
    add_paragraph(body, "Ngày kiểm thử: %s" % date.today().strftime("%d/%m/%Y"))
    add_paragraph(body, "Model: %s | Tập test: %s" % (args.model, args.test_dir))
    pipeline = "grayscale không outline" if args.no_outline else "grayscale + outline"
    add_paragraph(body, "Phương pháp: TFLite CNN 96 × 96, %s; dự đoán độc lập từng ảnh." % pipeline, after=180)

    add_paragraph(body, "1. Kết quả tổng quan", bold=True, size=26, color="17365D", before=100)
    accuracy = (100.0 * correct / total) if total else 0.0
    add_paragraph(body, "Accuracy: %d/%d = %.1f%%." % (correct, total, accuracy), bold=True, size=24)
    add_paragraph(body, "Số lớp: %d. Số ảnh: %d. Model được đánh giá: %s." % (len(LABELS), total, args.model))
    add_paragraph(body, "2. Số ảnh đúng theo nhãn", bold=True, size=26, color="17365D", before=140)
    rows = [["Nhãn đúng", "Tổng ảnh", "Đúng", "Sai", "Recall"]]
    for label in LABELS:
        n, c = per_class[label]
        rows.append([label, n, c, n-c, ("%.1f%%" % (100*c/n)) if n else "N/A"])
    rows.append(["Tổng", total, correct, total-correct, "%.1f%%" % accuracy])
    add_table(body, rows, [2300, 1500, 1300, 1300, 1500])
    add_paragraph(body, "3. Ma trận nhầm lẫn", bold=True, size=26, color="17365D", before=100)
    add_paragraph(body, "Hàng là nhãn đúng; cột là nhãn dự đoán. Số trong ô là số ảnh.")
    add_table(body, [["Đúng \\ Dự đoán"] + list(LABELS)] + [[label] + [matrix[label][pred] for pred in LABELS] for label in LABELS], [1800, 1250, 1250, 1250, 1600, 1250])
    add_paragraph(body, "4. Nhận xét và giới hạn", bold=True, size=26, color="17365D", before=100)
    if total and correct == total:
        add_paragraph(body, "Model phân loại đúng toàn bộ ảnh trong tập test này. Kết quả chỉ phản ánh tập ảnh hiện có; cần kiểm tra thêm ảnh độc lập và video từ đúng camera ESP32-S3 trước khi kết luận khả năng hoạt động thực tế.")
    elif total and correct == 5 and per_class.get("fist", (0, 0))[1] == 5 and all(per_class.get(label, (0, 0))[1] == 0 for label in LABELS if label != "fist"):
        add_paragraph(body, "Model chỉ nhận đúng lớp fist; tất cả mẫu thuộc bốn lớp còn lại đều bị dự đoán thành fist. Đây là dấu hiệu model CNN hiện tại không tổng quát tốt trên tập test này, vì vậy chưa phù hợp để tuyên bố nhận dạng ổn định.")
    else:
        add_paragraph(body, "Cần xem các nhãn bị nhầm trong ma trận để xác định lớp khó. Kết quả chỉ phản ánh tập test hiện có và chưa thay thế cho kiểm thử video trên camera ESP32-S3.")
    add_paragraph(body, "Tập test hiện có nhỏ (25 ảnh, 5 ảnh/lớp). Năm ảnh trong mỗi lớp có chung tiền tố tên tệp, có thể là các frame cùng một lần thu; do đó chúng không đại diện cho 25 tình huống độc lập. Không dùng tập test này để huấn luyện hoặc augmentation.")
    if Path(args.test_dir).name.lower() == "test_images":
        add_paragraph(body, "Đối chiếu SHA-256 cho thấy 25 ảnh của test_images trùng chính xác với các ảnh nằm trong test_hagrid (75 ảnh). Vì vậy hai thư mục không phải hai tập đánh giá độc lập; không cộng hai accuracy như hai thử nghiệm riêng.")
    add_paragraph(body, "Đây là kết quả của đúng tệp TFLite nêu trên. Firmware hiện vẫn ưu tiên model_outline.h và áp dụng outline; model thử nghiệm grayscale chưa được tích hợp vào firmware. Chưa đo độ trễ, FPS, RAM hoặc độ chính xác video trên board.")
    add_paragraph(body, "5. Danh sách ảnh và dự đoán", bold=True, size=26, color="17365D", before=100)
    add_paragraph(body, "Chi tiết từng ảnh được lưu riêng trong test_predictions.csv. Dùng CSV để lọc các trường hợp nhầm và bổ sung ghi chú khi hoàn thiện báo cáo.")
    add_paragraph(body, "Tệp nguồn: %s | Model: %s" % (args.test_dir, args.model), size=18, color="666666", before=180)
    ET.SubElement(body, q("sectPr"))

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
    doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'''
    core = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>Báo cáo kiểm thử demo nhận diện cử chỉ ESP32-S3</dc:title><dc:subject>Kết quả đánh giá model TFLite trên tập test</dc:subject><dc:creator>OpenAI</dc:creator></cp:coreProperties>'''
    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Word</Application></Properties>'''
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", ET.tostring(document, encoding="utf-8", xml_declaration=True))
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_dir", nargs="?", default="test_images")
    parser.add_argument("--model", default="model_outline.tflite")
    parser.add_argument("--report", default="reports/Bao_cao_kiem_thu_tflite.docx")
    parser.add_argument("--csv", default="reports/test_predictions.csv")
    parser.add_argument("--no-outline", action="store_true",
                        help="Đưa grayscale gốc vào model train không outline")
    args = parser.parse_args()
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.test_dir)
    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        parser.error("Không tìm thấy ảnh trong %s" % root)
    labeled = all(p.parent.name in LABELS for p in images)
    if not labeled:
        parser.error("Cần thư mục con đặt tên đúng theo nhãn: %s" % ", ".join(LABELS))

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    scale, zero_point = input_info["quantization"]
    matrix = {label: {pred: 0 for pred in LABELS} for label in LABELS}
    per_class = {label: [0, 0] for label in LABELS}
    rows = []
    for path in images:
        x = preprocess(path, no_outline=args.no_outline)
        if input_info["dtype"] in (np.int8, np.uint8):
            limits = np.iinfo(input_info["dtype"])
            x = np.clip(np.rint(x / scale + zero_point), limits.min, limits.max).astype(input_info["dtype"])
        else:
            x = x.astype(input_info["dtype"])
        interpreter.set_tensor(input_info["index"], x)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_info["index"])[0].astype(np.float32)
        out_scale, out_zero = output_info["quantization"]
        if output_info["dtype"] in (np.int8, np.uint8) and out_scale:
            scores = (scores - out_zero) * out_scale
        expected = path.parent.name
        predicted = LABELS[int(np.argmax(scores))]
        confidence = float(np.max(scores))
        ok = predicted == expected
        matrix[expected][predicted] += 1
        per_class[expected][0] += 1
        per_class[expected][1] += int(ok)
        rows.append((str(path), expected, predicted, "%.6f" % confidence, "yes" if ok else "no"))
        print("%s: predicted=%s expected=%s confidence=%.3f %s" % (path, predicted, expected, confidence, "OK" if ok else "WRONG"))
    total = len(rows)
    correct = sum(v[1] for v in per_class.values())
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "expected", "predicted", "confidence", "correct"])
        writer.writerows(rows)
    write_docx(args.report, args, rows, per_class, matrix, total, correct)
    print("ACCURACY %d/%d = %.1f%%" % (correct, total, 100.0 * correct / total))
    for label in LABELS:
        print("%s %d/%d" % (label, per_class[label][1], per_class[label][0]))
    print("REPORT %s" % Path(args.report).resolve())
    print("CSV %s" % Path(args.csv).resolve())


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    main()
