#!/usr/bin/env python3
"""Merge local, augmented, online and HaGRID train images without test leakage."""
import argparse
import csv
import hashlib
from pathlib import Path
import shutil
import zipfile

LABELS = ("fist", "palm", "peace", "thumbs_up", "none")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HAGRID_LABELS = {"fist": "fist", "none": "none", "open": "palm",
                 "peace": "peace", "thumb_up": "thumbs_up"}


def image_files(folder):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dataset_train_complete")
    parser.add_argument("--hagrid-zip", default="tinyml_improved_20261016.zip",
                        help="ZIP HaGRID đặt cạnh dự án (mặc định: tinyml_improved_20261016.zip)")
    parser.add_argument("--zip-out", default="dataset_train_all.zip")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    output = root / args.out
    if output.exists():
        if not args.force:
            parser.error("{} đã tồn tại; dùng --force để tạo lại".format(output))
        shutil.rmtree(output)
    output.mkdir(parents=True)
    sources = []
    for label in LABELS:
        local_labels = ("other", "none") if label == "none" else (label,)
        for local_label in local_labels:
            for path in image_files(root / "dataset" / local_label):
                sources.append((path, label, "local", "dataset/" + local_label, path.name))
        online = root / "dataset_online" / "shimaazizi" / label
        for path in image_files(online):
            sources.append((path, label, "online", online.as_posix(), path.name))
        for version, dirname in (("v1", "dataset_augmented"), ("v2", "dataset_augmented_v2")):
            folder = root / dirname / label
            for path in sorted(folder.glob("aug__*")) if folder.is_dir() else []:
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    sources.append((path, label, "augmented", dirname + "/" + label,
                                    version + "__" + path.name))

    zip_path = Path(args.hagrid_zip)
    if not zip_path.is_file():
        parser.error("Không tìm thấy ZIP dữ liệu: {}".format(zip_path))
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/")
            # Use only the official train split. Validation and test remain untouched.
            if (len(parts) >= 5 and parts[:3] == ["dataset", "poses", "train"]
                    and parts[3] in HAGRID_LABELS and Path(parts[-1]).suffix.lower() in EXTENSIONS):
                label = HAGRID_LABELS[parts[3]]
                sources.append((info, label, "hagrid_train", info.filename, parts[-1]))

    hashes = set()
    counts = {label: {kind: 0 for kind in ("local", "augmented", "online", "hagrid_train")}
              for label in LABELS}
    manifest_path = output / "MANIFEST.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["label", "source_type", "source_path", "source_filename", "train_filename", "sha256"])
        hagrid = zipfile.ZipFile(zip_path)
        try:
            for source, label, source_type, source_path, source_filename in sources:
                if source_type == "hagrid_train":
                    data = hagrid.read(source)
                    suffix = Path(source_filename).suffix.lower()
                else:
                    data = source.read_bytes()
                    suffix = source.suffix.lower()
                digest = hashlib.sha256(data).hexdigest()
                key = (label, digest)
                if key in hashes:
                    continue
                hashes.add(key)
                # Preserve the original timestamp/id in augmentation names so the
                # trainer can keep each source image with its generated variants.
                if source_type == "augmented":
                    version, name = source_filename.split("__", 1)
                    stem = Path(name).stem
                    if stem.startswith("aug__"):
                        stem = stem[5:]
                    bits = stem.rsplit("__", 1)
                    stem = "aug__{}__{}__{}".format(bits[0], version, bits[1]) if len(bits) == 2 else "aug__" + stem
                elif source_type == "local":
                    stem = Path(source_filename).stem
                elif source_type == "online":
                    stem = "online__" + Path(source_filename).stem
                else:
                    stem = "hagrid__" + Path(source_filename).stem
                filename = "{}{}".format(stem, suffix)
                destination = output / label / filename
                if destination.exists():
                    filename = "{}__{}{}".format(stem, digest[:10], suffix)
                    destination = output / label / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                counts[label][source_type] += 1
                writer.writerow([label, source_type, source_path, source_filename, filename, digest])
        finally:
            hagrid.close()

    summary = ["CNN training data source totals", "Only HaGRID train split is included; its val/test stay excluded.", ""]
    total = 0
    for label in LABELS:
        values = counts[label]
        subtotal = sum(values.values())
        total += subtotal
        summary.append("{}: local={}, augmented={}, online={}, hagrid_train={}, total={}".format(
            label, values["local"], values["augmented"], values["online"], values["hagrid_train"], subtotal))
    summary.append("TOTAL: {}".format(total))
    (output / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("TRAIN_DATA", output.resolve())
    print("TRAIN_IMAGES", total)
    for line in summary[2:-1]:
        print(line)

    # Extract only HaGRID test images to a separate, untouched evaluation set.
    test_output = root / "test_hagrid"
    if test_output.exists():
        shutil.rmtree(test_output)
    test_output.mkdir()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            parts = info.filename.replace("\\", "/").split("/")
            if (len(parts) >= 5 and parts[:3] == ["dataset", "poses", "test"]
                    and parts[3] in HAGRID_LABELS and Path(parts[-1]).suffix.lower() in EXTENSIONS):
                label = HAGRID_LABELS[parts[3]]
                target = test_output / label / Path(parts[-1]).name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print("HAGRID_TEST_IMAGES", sum(len(image_files(test_output / label)) for label in LABELS))

    archive_path = root / args.zip_out
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as merged:
        for label in LABELS:
            for path in image_files(output / label):
                merged.write(path, "dataset/{}/{}".format(label, path.name))
        merged.write(manifest_path, "MANIFEST.csv")
        merged.write(output / "SUMMARY.txt", "SUMMARY.txt")
        for name in ("SOURCE.txt", "LICENSE-MIT.txt"):
            license_path = root / "dataset_online" / "shimaazizi" / name
            if license_path.is_file():
                merged.write(license_path, name)
        merged.writestr("HAGRID_SOURCE.txt",
                        "HaGRID train images were taken from tinyml_improved_20261016.zip. "
                        "Only the train split is included here; the archive val and test splits are excluded.\n")
    print("TRAIN_ARCHIVE", archive_path.resolve(), archive_path.stat().st_size)


if __name__ == "__main__":
    main()
