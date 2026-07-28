import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def read_text_auto_encoding(file_path: Path) -> str:
    """
    尝试使用常见编码读取文本文件。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件：{file_path}")

    encodings = (
        "utf-8",
        "utf-8-sig",
        "gbk",
        "gb18030",
    )

    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"无法识别文件编码：{file_path}")


def normalize_newlines(text: str) -> str:
    """
    统一换行符。
    """
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def load_imgids(file_path: Path) -> List[int]:
    """
    从固定划分文件中读取IMGID。

    支持格式：
        IMGID:1093
        IMGID：1093
    """
    text = read_text_auto_encoding(file_path)

    matched_ids = re.findall(
        r"IMGID\s*[:：]\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if not matched_ids:
        raise ValueError(
            f"{file_path} 中没有找到任何IMGID"
        )

    imgids = [int(imgid) for imgid in matched_ids]

    counts = Counter(imgids)
    duplicate_ids = sorted(
        imgid
        for imgid, count in counts.items()
        if count > 1
    )

    if duplicate_ids:
        raise ValueError(
            f"{file_path} 内部存在重复IMGID："
            f"{duplicate_ids}"
        )

    return imgids


def validate_splits(
    train_ids: List[int],
    valid_ids: List[int],
    test_ids: List[int],
) -> List[int]:
    """
    检查训练集、验证集和测试集是否相互独立，
    并返回全部保留样本ID。
    """
    train_set = set(train_ids)
    valid_set = set(valid_ids)
    test_set = set(test_ids)

    overlap_train_valid = train_set & valid_set
    overlap_train_test = train_set & test_set
    overlap_valid_test = valid_set & test_set

    if overlap_train_valid:
        raise ValueError(
            "训练集与验证集存在重复IMGID："
            f"{sorted(overlap_train_valid)}"
        )

    if overlap_train_test:
        raise ValueError(
            "训练集与测试集存在重复IMGID："
            f"{sorted(overlap_train_test)}"
        )

    if overlap_valid_test:
        raise ValueError(
            "验证集与测试集存在重复IMGID："
            f"{sorted(overlap_valid_test)}"
        )

    retained_ids = sorted(
        train_set | valid_set | test_set
    )

    return retained_ids


def write_retained_ids(
    retained_ids: List[int],
    output_file: Path,
) -> None:
    """
    将全部保留样本ID按照数字升序写入文件。
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = "\n".join(
        f"IMGID:{imgid}"
        for imgid in retained_ids
    )

    output_file.write_text(
        content + "\n",
        encoding="utf-8",
    )


def parse_annotation_file(
    file_path: Path,
) -> List[Tuple[int, str]]:
    """
    从一个文本标注文件中解析完整样本。

    每个样本格式示例：

        IMGID:1093
        叶    O
        片    O
        晚    B-Disease
        疫    I-Disease
        病    I-Disease
    """
    text = read_text_auto_encoding(file_path)
    text = normalize_newlines(text).strip()

    if not text:
        return []

    blocks = re.split(
        r"(?=^IMGID\s*[:：]\s*\d+\s*$)",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    samples: List[Tuple[int, str]] = []

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        match = re.match(
            r"^IMGID\s*[:：]\s*(\d+)",
            block,
            flags=re.IGNORECASE,
        )

        if not match:
            # 忽略不以IMGID开头的说明性内容
            continue

        imgid = int(match.group(1))

        # 统一首行为IMGID:数字
        normalized_block = re.sub(
            r"^IMGID\s*[:：]\s*\d+",
            f"IMGID:{imgid}",
            block,
            count=1,
            flags=re.IGNORECASE,
        )

        samples.append(
            (imgid, normalized_block)
        )

    return samples


def build_text_sample_index(
    text_data_dir: Path,
) -> Dict[int, str]:
    """
    读取文本目录中的所有txt文件，
    建立IMGID到完整文本及BIO标注的映射。
    """
    if not text_data_dir.exists():
        raise FileNotFoundError(
            f"找不到文本数据目录：{text_data_dir}"
        )

    txt_files = sorted(
        text_data_dir.rglob("*.txt")
    )

    if not txt_files:
        raise FileNotFoundError(
            f"{text_data_dir} 中没有找到txt文件"
        )

    sample_index: Dict[int, str] = {}
    sample_sources: Dict[int, Path] = {}

    for file_path in txt_files:
        samples = parse_annotation_file(
            file_path
        )

        for imgid, sample_text in samples:
            if imgid in sample_index:
                raise ValueError(
                    f"文本标注中IMGID:{imgid}重复出现。\n"
                    f"第一次出现：{sample_sources[imgid]}\n"
                    f"第二次出现：{file_path}"
                )

            sample_index[imgid] = sample_text
            sample_sources[imgid] = file_path

    if not sample_index:
        raise ValueError(
            f"{text_data_dir} 中没有解析出任何IMGID样本"
        )

    return sample_index


def check_bio_sequence(
    imgid: int,
    sample_text: str,
) -> List[str]:
    """
    对文本样本进行基本BIO合法性检查。

    检查内容：
    1. 每个token行是否至少有token和标签；
    2. 标签是否为O、B-类型或I-类型；
    3. I-类型前面是否为同类型的B或I。

    返回错误信息列表。
    """
    errors: List[str] = []
    previous_label = "O"

    lines = sample_text.splitlines()

    for line_number, line in enumerate(
        lines[1:],
        start=2,
    ):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 2:
            errors.append(
                f"IMGID:{imgid} 第{line_number}行"
                f"缺少BIO标签：{line}"
            )
            continue

        label = parts[-1]

        if label == "O":
            previous_label = "O"
            continue

        label_match = re.fullmatch(
            r"([BI])-(.+)",
            label,
        )

        if not label_match:
            errors.append(
                f"IMGID:{imgid} 第{line_number}行"
                f"标签格式错误：{label}"
            )
            previous_label = label
            continue

        prefix = label_match.group(1)
        entity_type = label_match.group(2)

        if prefix == "I":
            allowed_previous = {
                f"B-{entity_type}",
                f"I-{entity_type}",
            }

            if previous_label not in allowed_previous:
                errors.append(
                    f"IMGID:{imgid} 第{line_number}行"
                    f"存在非法BIO转移："
                    f"{previous_label} -> {label}"
                )

        previous_label = label

    return errors


def build_main_image_index(
    main_image_dir: Path,
) -> Dict[int, Path]:
    """
    建立IMGID到主图片路径的映射。

    适用于：
        data/Disease7000-Refined_images/1.jpg
        data/Disease7000-Refined_images/1093.jpg
    """
    if not main_image_dir.exists():
        raise FileNotFoundError(
            f"找不到主图片目录：{main_image_dir}"
        )

    image_index: Dict[int, Path] = {}

    for image_path in sorted(
        main_image_dir.rglob("*")
    ):
        if not image_path.is_file():
            continue

        if (
            image_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        # 主图片文件名应为纯数字，例如1.jpg
        if not image_path.stem.isdigit():
            print(
                "警告：跳过无法识别IMGID的主图片："
                f"{image_path}"
            )
            continue

        imgid = int(image_path.stem)

        if imgid in image_index:
            raise ValueError(
                f"IMGID:{imgid}对应多张主图片：\n"
                f"{image_index[imgid]}\n"
                f"{image_path}"
            )

        image_index[imgid] = image_path

    if not image_index:
        raise ValueError(
            f"{main_image_dir} 中没有找到有效主图片"
        )

    return image_index


def parse_aux_image_name(
    image_path: Path,
) -> Optional[Tuple[int, int]]:
    """
    从辅助图片名中解析IMGID和crop编号。

    示例：
        1_pred_yolo_crop_0.png
        1093_pred_yolo_crop_2.png

    返回：
        (IMGID, crop_index)
    """
    match = re.fullmatch(
        r"(\d+)_pred_yolo_crop_(\d+)",
        image_path.stem,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    imgid = int(match.group(1))
    crop_index = int(match.group(2))

    return imgid, crop_index


def build_aux_image_index(
    aux_image_dir: Path,
) -> Dict[int, List[Path]]:
    """
    建立IMGID到辅助图片列表的映射。

    支持递归读取：

        data/Disease7000-Refined_aux_images/
            train/crops/1_pred_yolo_crop_0.png
            valid/crops/2_pred_yolo_crop_0.png
            test/crops/3_pred_yolo_crop_1.png
    """
    if not aux_image_dir.exists():
        raise FileNotFoundError(
            f"找不到辅助图片目录：{aux_image_dir}"
        )

    temp_index: Dict[
        int,
        List[Tuple[int, Path]]
    ] = defaultdict(list)

    crop_keys: Dict[
        Tuple[int, int],
        Path
    ] = {}

    for image_path in sorted(
        aux_image_dir.rglob("*")
    ):
        if not image_path.is_file():
            continue

        if (
            image_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        parsed = parse_aux_image_name(
            image_path
        )

        if parsed is None:
            print(
                "警告：跳过无法识别IMGID的辅助图片："
                f"{image_path}"
            )
            continue

        imgid, crop_index = parsed
        key = (imgid, crop_index)

        if key in crop_keys:
            raise ValueError(
                f"IMGID:{imgid}的crop_{crop_index}"
                f"存在重复辅助图片：\n"
                f"{crop_keys[key]}\n"
                f"{image_path}"
            )

        crop_keys[key] = image_path
        temp_index[imgid].append(
            (crop_index, image_path)
        )

    aux_index: Dict[int, List[Path]] = {}

    for imgid, items in temp_index.items():
        sorted_items = sorted(
            items,
            key=lambda item: item[0],
        )

        aux_index[imgid] = [
            image_path
            for _, image_path in sorted_items
        ]

    return aux_index


def write_text_split(
    split_name: str,
    imgids: List[int],
    text_samples: Dict[int, str],
    output_dir: Path,
) -> None:
    """
    根据固定划分ID生成完整文本及BIO标注文件。
    """
    missing_ids = [
        imgid
        for imgid in imgids
        if imgid not in text_samples
    ]

    if missing_ids:
        raise ValueError(
            f"{split_name}划分中缺少以下文本样本："
            f"{missing_ids}"
        )

    output_blocks = [
        text_samples[imgid]
        for imgid in imgids
    ]

    output_text_dir = (
        output_dir / "Disease7000-Refined"
    )
    output_text_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_text_dir / f"{split_name}.txt"
    )

    output_file.write_text(
        "\n\n".join(output_blocks) + "\n",
        encoding="utf-8",
    )


def copy_main_images(
    split_name: str,
    imgids: List[int],
    main_image_index: Dict[int, Path],
    output_dir: Path,
) -> None:
    """
    根据固定划分ID复制主图片。
    """
    missing_ids = [
        imgid
        for imgid in imgids
        if imgid not in main_image_index
    ]

    if missing_ids:
        raise ValueError(
            f"{split_name}划分中缺少以下主图片："
            f"{missing_ids}"
        )

    output_image_dir = (
        output_dir
        / "Disease7000-Refined_images"
        / split_name
    )

    output_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for imgid in imgids:
        source_image = main_image_index[imgid]

        target_image = (
            output_image_dir
            / source_image.name
        )

        shutil.copy2(
            source_image,
            target_image,
        )


def copy_aux_images(
    split_name: str,
    imgids: List[int],
    aux_image_index: Dict[int, List[Path]],
    output_dir: Path,
) -> List[int]:
    """
    根据固定划分ID复制辅助图片。

    辅助图片缺失时不会停止程序，
    而是记录缺失IMGID并继续运行。
    """
    output_aux_dir = (
        output_dir
        / "Disease7000-Refined_aux_images"
        / split_name
        / "crops"
    )

    output_aux_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    missing_aux_ids: List[int] = []

    for imgid in imgids:
        source_images = aux_image_index.get(
            imgid,
            [],
        )

        if not source_images:
            missing_aux_ids.append(imgid)
            continue

        for source_image in source_images:
            target_image = (
                output_aux_dir
                / source_image.name
            )

            shutil.copy2(
                source_image,
                target_image,
            )

    return missing_aux_ids


def reconstruct_split(
    split_name: str,
    imgids: List[int],
    text_samples: Dict[int, str],
    main_image_index: Dict[int, Path],
    aux_image_index: Dict[int, List[Path]],
    output_dir: Path,
) -> List[int]:
    """
    使用同一组IMGID同步重建文本、主图片和辅助图片。
    """
    write_text_split(
        split_name=split_name,
        imgids=imgids,
        text_samples=text_samples,
        output_dir=output_dir,
    )

    copy_main_images(
        split_name=split_name,
        imgids=imgids,
        main_image_index=main_image_index,
        output_dir=output_dir,
    )

    missing_aux_ids = copy_aux_images(
        split_name=split_name,
        imgids=imgids,
        aux_image_index=aux_image_index,
        output_dir=output_dir,
    )

    print(
        f"{split_name}：成功重建"
        f"{len(imgids)}个文本—图片样本"
    )

    if missing_aux_ids:
        print(
            f"{split_name}："
            f"{len(missing_aux_ids)}个样本"
            "没有辅助图片"
        )

    return missing_aux_ids


def write_report(
    report_file: Path,
    train_ids: List[int],
    valid_ids: List[int],
    test_ids: List[int],
    retained_ids: List[int],
    text_samples: Dict[int, str],
    main_image_index: Dict[int, Path],
    aux_image_index: Dict[int, List[Path]],
    missing_aux_by_split: Dict[str, List[int]],
) -> None:
    """
    保存数据重建与验证报告。
    """
    total_aux_images = sum(
        len(paths)
        for paths in aux_image_index.values()
    )

    report_lines = [
        "Disease7000-Refined reconstruction report",
        "=" * 60,
        f"Training IDs: {len(train_ids)}",
        f"Validation IDs: {len(valid_ids)}",
        f"Test IDs: {len(test_ids)}",
        f"Total retained IDs: {len(retained_ids)}",
        "",
        f"Parsed text samples: {len(text_samples)}",
        f"Indexed main images: {len(main_image_index)}",
        (
            "Samples with auxiliary images: "
            f"{len(aux_image_index)}"
        ),
        (
            "Total auxiliary images: "
            f"{total_aux_images}"
        ),
        "",
        "Split overlap check: passed",
        "Retained ID union check: passed",
        "Text availability check: passed",
        "Main-image availability check: passed",
    ]

    for split_name in (
        "train",
        "valid",
        "test",
    ):
        missing_ids = missing_aux_by_split.get(
            split_name,
            [],
        )

        report_lines.append(
            f"{split_name} samples without "
            f"auxiliary images: {len(missing_ids)}"
        )

        if missing_ids:
            report_lines.append(
                ", ".join(
                    f"IMGID:{imgid}"
                    for imgid in missing_ids
                )
            )

    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the fixed multimodal dataset "
            "split using released IMGID manifests."
        )
    )

    parser.add_argument(
        "--text-dir",
        type=Path,
        default=Path("data/Disease7000-Refined"),
        help=(
            "原始文本及BIO标注目录，默认："
            "data/v"
        ),
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(
            "data/Disease7000-Refined_images"
        ),
        help=(
            "原始主图片目录，默认："
            "data/Disease7000-Refined_images"
        ),
    )

    parser.add_argument(
        "--aux-image-dir",
        type=Path,
        default=Path(
            "data/Disease7000-Refined_aux_images"
        ),
        help=(
            "原始辅助图片目录，默认："
            "data/v_aux_images"
        ),
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data_yuchuli"),
        help=(
            "固定划分ID目录，默认："
            "data_yuchuli"
        ),
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("manifests"),
        help=(
            "总保留ID输出目录，默认："
            "manifests"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reconstructed_data"
        ),
        help=(
            "重建数据输出目录，默认："
            "reconstructed_data"
        ),
    )

    parser.add_argument(
        "--skip-bio-check",
        action="store_true",
        help="跳过BIO合法性检查",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有输出目录",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    train_ids_file = (
        args.split_dir / "train.txt"
    )
    valid_ids_file = (
        args.split_dir / "valid.txt"
    )
    test_ids_file = (
        args.split_dir / "test.txt"
    )

    retained_ids_file = (
        args.manifest_dir
        / "retained_sample_ids.txt"
    )

    report_file = (
        args.manifest_dir
        / "reconstruction_report.txt"
    )

    if args.output_dir.exists():
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            raise FileExistsError(
                f"输出目录已存在：{args.output_dir}\n"
                "如需覆盖，请添加 --overwrite"
            )

    print("=" * 60)
    print("一、读取固定划分ID")
    print("=" * 60)

    train_ids = load_imgids(
        train_ids_file
    )
    valid_ids = load_imgids(
        valid_ids_file
    )
    test_ids = load_imgids(
        test_ids_file
    )

    print(f"训练集ID数量：{len(train_ids)}")
    print(f"验证集ID数量：{len(valid_ids)}")
    print(f"测试集ID数量：{len(test_ids)}")

    retained_ids = validate_splits(
        train_ids=train_ids,
        valid_ids=valid_ids,
        test_ids=test_ids,
    )

    print(
        f"三个划分合并后的唯一IMGID数量："
        f"{len(retained_ids)}"
    )
    print("三个划分之间不存在重复IMGID。")

    write_retained_ids(
        retained_ids=retained_ids,
        output_file=retained_ids_file,
    )

    print(
        f"总保留ID已保存到："
        f"{retained_ids_file.resolve()}"
    )

    print("\n" + "=" * 60)
    print("二、读取文本及BIO标注")
    print("=" * 60)

    text_samples = build_text_sample_index(
        args.text_dir
    )

    print(
        f"共解析到文本样本："
        f"{len(text_samples)}"
    )

    retained_set = set(retained_ids)
    missing_text_ids = sorted(
        retained_set - set(text_samples)
    )

    if missing_text_ids:
        raise ValueError(
            "以下保留IMGID缺少文本或BIO标注："
            f"{missing_text_ids}"
        )

    if not args.skip_bio_check:
        print("开始检查保留样本BIO标签……")

        bio_errors: List[str] = []

        for imgid in retained_ids:
            errors = check_bio_sequence(
                imgid=imgid,
                sample_text=text_samples[imgid],
            )
            bio_errors.extend(errors)

        if bio_errors:
            print("\n发现BIO标签问题：")

            for error in bio_errors:
                print(error)

            raise ValueError(
                f"共发现{len(bio_errors)}个BIO问题，"
                "停止重建。"
            )

        print("保留样本BIO标签检查通过。")
    else:
        print("已跳过BIO合法性检查。")

    print("\n" + "=" * 60)
    print("三、读取主图片")
    print("=" * 60)

    main_image_index = build_main_image_index(
        args.image_dir
    )

    print(
        f"共索引到主图片："
        f"{len(main_image_index)}"
    )

    missing_main_image_ids = sorted(
        retained_set - set(main_image_index)
    )

    if missing_main_image_ids:
        raise ValueError(
            "以下保留IMGID缺少主图片："
            f"{missing_main_image_ids}"
        )

    print("所有保留样本均具有对应主图片。")

    print("\n" + "=" * 60)
    print("四、读取辅助图片")
    print("=" * 60)

    aux_image_index = build_aux_image_index(
        args.aux_image_dir
    )

    total_aux_images = sum(
        len(paths)
        for paths in aux_image_index.values()
    )

    print(
        "具有辅助图片的样本数量："
        f"{len(aux_image_index)}"
    )
    print(
        f"辅助图片总数：{total_aux_images}"
    )

    print("\n" + "=" * 60)
    print("五、统一重建文本、主图片和辅助图片")
    print("=" * 60)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    missing_aux_by_split: Dict[
        str,
        List[int]
    ] = {}

    missing_aux_by_split["train"] = (
        reconstruct_split(
            split_name="train",
            imgids=train_ids,
            text_samples=text_samples,
            main_image_index=main_image_index,
            aux_image_index=aux_image_index,
            output_dir=args.output_dir,
        )
    )

    missing_aux_by_split["valid"] = (
        reconstruct_split(
            split_name="valid",
            imgids=valid_ids,
            text_samples=text_samples,
            main_image_index=main_image_index,
            aux_image_index=aux_image_index,
            output_dir=args.output_dir,
        )
    )

    missing_aux_by_split["test"] = (
        reconstruct_split(
            split_name="test",
            imgids=test_ids,
            text_samples=text_samples,
            main_image_index=main_image_index,
            aux_image_index=aux_image_index,
            output_dir=args.output_dir,
        )
    )

    write_report(
        report_file=report_file,
        train_ids=train_ids,
        valid_ids=valid_ids,
        test_ids=test_ids,
        retained_ids=retained_ids,
        text_samples=text_samples,
        main_image_index=main_image_index,
        aux_image_index=aux_image_index,
        missing_aux_by_split=(
            missing_aux_by_split
        ),
    )

    print("\n" + "=" * 60)
    print("重建完成")
    print("=" * 60)

    print(f"训练集样本数：{len(train_ids)}")
    print(f"验证集样本数：{len(valid_ids)}")
    print(f"测试集样本数：{len(test_ids)}")
    print(f"全部保留样本数：{len(retained_ids)}")

    print(
        f"重建数据目录："
        f"{args.output_dir.resolve()}"
    )
    print(
        f"重建报告："
        f"{report_file.resolve()}"
    )


if __name__ == "__main__":
    main()