from pathlib import Path

import yaml

import scripts.split_livingroom_dirichlet as splitter
from scripts.split_livingroom_dirichlet import load_samples, normalize_source_dir


def write_image_and_label(root, split, image_name, class_id):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / image_name).write_bytes(b"synthetic image")
    (label_dir / Path(image_name).with_suffix(".txt").name).write_text(
        f"{class_id} 0.5 0.5 0.25 0.25\n", encoding="utf-8"
    )


def test_load_samples_supports_converted_voc_layout(tmp_path):
    voc_root = tmp_path / "VOC"
    names = {0: "aeroplane", 1: "bicycle"}
    (tmp_path / "VOC.yaml").write_text(
        yaml.safe_dump({"nc": 2, "names": names}), encoding="utf-8"
    )
    write_image_and_label(voc_root, "train2007", "2007_000001.jpg", 0)
    write_image_and_label(voc_root, "train2012", "2012_000001.jpg", 1)
    write_image_and_label(voc_root, "val2007", "2007_000002.jpg", 0)
    write_image_and_label(voc_root, "val2012", "2012_000002.jpg", 1)
    write_image_and_label(voc_root, "test2007", "2007_000003.jpg", 1)

    samples, val_images, val_labels, class_names = load_samples(
        voc_root / "images"
    )

    assert normalize_source_dir(voc_root / "images") == voc_root
    assert [sample.image.name for sample in samples] == [
        "2007_000001.jpg",
        "2007_000002.jpg",
        "2012_000001.jpg",
        "2012_000002.jpg",
    ]
    assert [sample.class_ids for sample in samples] == [(0,), (0,), (1,), (1,)]
    assert [path.name for path in val_images] == ["2007_000003.jpg"]
    assert len(val_labels) == 1
    assert class_names == ["aeroplane", "bicycle"]


def test_load_samples_preserves_original_flat_yolo_layout(tmp_path):
    root = tmp_path / "livingroom"
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images" / f"{split}.jpg").write_bytes(b"image")
        (root / split / "labels" / f"{split}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    (root / "dataset.yaml").write_text(
        yaml.safe_dump({"nc": 1, "names": ["object"]}), encoding="utf-8"
    )

    samples, val_images, val_labels, class_names = load_samples(root)

    assert len(samples) == 1
    assert len(val_images) == 1
    assert len(val_labels) == 1
    assert class_names == ["object"]


def test_voc_split_end_to_end_writes_yolo_client_datasets(monkeypatch, tmp_path):
    voc_root = tmp_path / "VOC"
    output = tmp_path / "output"
    (tmp_path / "VOC.yaml").write_text(
        yaml.safe_dump({"nc": 2, "names": {0: "aeroplane", 1: "bicycle"}}),
        encoding="utf-8",
    )
    for index in range(4):
        write_image_and_label(
            voc_root,
            "train2007" if index < 2 else "train2012",
            f"train_{index}.jpg",
            index % 2,
        )
    write_image_and_label(voc_root, "val2007", "val_2007.jpg", 0)
    write_image_and_label(voc_root, "val2012", "val_2012.jpg", 1)
    write_image_and_label(voc_root, "test2007", "test_2007.jpg", 1)
    monkeypatch.setattr(
        "sys.argv",
        [
            "split_livingroom_dirichlet.py",
            "--source",
            str(voc_root / "images"),
            "--output",
            str(output),
            "--num-clients",
            "2",
            "--mode",
            "iid",
            "--seed",
            "3",
        ],
    )

    splitter.main()

    assert len(list((output / "client_1" / "train" / "images").glob("*.jpg"))) == 3
    assert len(list((output / "client_2" / "train" / "images").glob("*.jpg"))) == 3
    assert len(list((output / "client_1" / "valid" / "images").glob("*.jpg"))) == 1
    assert len(list((output / "client_2" / "valid" / "images").glob("*.jpg"))) == 1
    client_yaml = yaml.safe_load((output / "client_1.yaml").read_text())
    assert client_yaml["nc"] == 2
    assert client_yaml["names"] == {0: "aeroplane", 1: "bicycle"}
    assert (output / "class_distribution_counts.csv").is_file()
    assert (output / "class_distribution_heatmap.png").is_file()
