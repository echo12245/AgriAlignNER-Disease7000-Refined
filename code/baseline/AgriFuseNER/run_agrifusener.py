from __future__ import annotations
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from processor.dataset import (
    Disease7000RefinedDataset,
    ENTITY_TYPES,
    build_label_vocab,
    collate_batch,
    parse_imgid_bio_file,
)
from models.agrifusener import AgriFuseNERApprox
from utils.evaluate import decode_word_level, metrics
from utils.visual import extract_split_features
DATA_DIR = Path("data/Disease7000-Refined")
IMAGE_DIR = Path("data/Disease7000-Refined_images")
AUX_ROOT = Path("data/Disease7000-Refined_aux_images")
BERT_MODEL_NAME = "bert-base-cased"
VIT_MODEL_NAME = "google/vit-base-patch16-224"
TRAIN_FILE = DATA_DIR / "train.txt"
DEV_FILE = DATA_DIR / "valid.txt"
TEST_FILE = DATA_DIR / "test.txt"
TRAIN_AUX_DICT = DATA_DIR / "Disease7000-Refined_train_dict.pth"
DEV_AUX_DICT = DATA_DIR / "Disease7000-Refined_val_dict.pth"
TEST_AUX_DICT = DATA_DIR / "Disease7000-Refined_test_dict.pth"
TRAIN_CROP_DIR = AUX_ROOT / "train" / "crops"
DEV_CROP_DIR = AUX_ROOT / "val" / "crops"
TEST_CROP_DIR = AUX_ROOT / "test" / "crops"
EXPECTED_COUNTS = {"train": 5038, "dev": 629, "test": 631}
PAPER_EPOCHS = 120
PAPER_BATCH_SIZE = 32
PAPER_LR = 3e-6
PAPER_DROPOUT = 0.1
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def resolve_paths(args):
    data_dir = Path(args.data_dir)
    aux_root = Path(args.aux_root)
    feature_root = Path(args.feature_root)
    return {
        "train_file": Path(args.train_file or data_dir / "train.txt"),
        "dev_file": Path(args.dev_file or data_dir / "valid.txt"),
        "test_file": Path(args.test_file or data_dir / "test.txt"),
        "image_dir": Path(args.image_dir),
        "feature_root": feature_root,
        "train_aux_dict": Path(args.train_aux_dict or data_dir / "Disease7000-Refined_train_dict.pth"),
        "dev_aux_dict": Path(args.dev_aux_dict or data_dir / "Disease7000-Refined_val_dict.pth"),
        "test_aux_dict": Path(args.test_aux_dict or data_dir / "Disease7000-Refined_test_dict.pth"),
        "train_crop_dir": Path(args.train_crop_dir or aux_root / "train" / "crops"),
        "dev_crop_dir": Path(args.dev_crop_dir or aux_root / "val" / "crops"),
        "test_crop_dir": Path(args.test_crop_dir or aux_root / "test" / "crops"),
    }
def validate_data(paths, strict_counts=True):
    print("\n=== Disease7000-Refined validation ===")
    for split in ["train", "dev", "test"]:
        rows = parse_imgid_bio_file(paths[f"{split}_file"])
        print(f"{split:>5}: {len(rows)} samples -> {paths[f'{split}_file']}")
        if strict_counts and len(rows) != EXPECTED_COUNTS[split]:
            raise RuntimeError(
                f"{split} count is {len(rows)}, expected {EXPECTED_COUNTS[split]}. "
                "Use --no-strict-counts only for debugging/subsets."
            )
    print("Entity types:", ENTITY_TYPES)
def extract_all(paths, args):
    for split in ["train", "dev", "test"]:
        out_dir = paths["feature_root"] / split
        extract_split_features(
            text_file=paths[f"{split}_file"],
            image_dir=paths["image_dir"],
            aux_dict=paths[f"{split}_aux_dict"],
            crop_dir=paths[f"{split}_crop_dir"],
            out_dir=out_dir,
            vit_name=args.vit_name,
            top_k=args.top_k,
            device=args.device,
            local_files_only=args.local_files_only,
        )
def evaluate(model, loader, device, id2label):
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["visual_features"],
                batch["visual_mask"],
                batch["labels"],
            )
            y_true, y_pred = decode_word_level(
                out["ner_paths"],
                batch["labels"],
                batch["attention_mask"],
                batch["word_start_mask"],
                id2label,
            )
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
    return metrics(y_true_all, y_pred_all)
def load_model_state(checkpoint, device):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    try:
        return torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location=device)
def build_model_for_dataset(dataset, label2id, id2label, args, device):
    visual_dim = dataset[0]["visual_features"].shape[-1]
    model = AgriFuseNERApprox(
        num_tags=len(label2id),
        id2label=id2label,
        text_encoder=args.bert_name,
        visual_dim=visual_dim,
        align_dim=args.align_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        lambda_paper=args.lambda_paper,
        grouping_source_train=args.grouping_source_train,
        local_files_only=args.local_files_only,
    ).to(device)
    return model
def train_one(paths, args, out_dir=None):
    seed_all(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    label2id, id2label = build_label_vocab()
    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_name,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    def make_dataset(split):
        return Disease7000RefinedDataset(
            paths[f"{split}_file"],
            paths["feature_root"] / split,
            tokenizer,
            label2id,
            args.max_seq,
        )
    train_ds, dev_ds, test_ds = make_dataset("train"), make_dataset("dev"), make_dataset("test")
    if not args.no_strict_counts:
        for split, ds in [("train", train_ds), ("dev", dev_ds), ("test", test_ds)]:
            if len(ds) != EXPECTED_COUNTS[split]:
                raise RuntimeError(f"{split}: {len(ds)} samples, expected {EXPECTED_COUNTS[split]}")
    collate = lambda batch: collate_batch(batch, tokenizer.pad_token_id or 0, label2id["O"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = build_model_for_dataset(
        train_ds,
        label2id,
        id2label,
        args,
        device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    out_dir = Path(out_dir or args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "status": "approximate reimplementation of Huang et al. on Disease7000-Refined",
        "seed": args.seed,
        "lambda_paper": args.lambda_paper,
        "grouping_source_train": args.grouping_source_train,
        "bert_name": args.bert_name,
        "vit_name": args.vit_name,
        "local_files_only": args.local_files_only,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "entity_types": ENTITY_TYPES,
        "roi_substitution": "same ROI proposals as AgriAlignNER; ROI/global features encoded with frozen ViT",
    }
    (out_dir / "resolved_settings.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    best_dev_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in bar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["visual_features"],
                batch["visual_mask"],
                batch["labels"],
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            bar.set_postfix(loss=f"{out['loss'].item():.4f}")

        dev_metrics = evaluate(model, dev_loader, device, id2label)
        history.append(
            {
                "epoch": epoch,
                "dev_f1": dev_metrics["f1"],
                "dev_macro_f1": dev_metrics["macro_f1"],
            }
        )
        print(
            f"epoch={epoch} dev_F1={dev_metrics['f1']:.4f} "
            f"dev_macroF1={dev_metrics['macro_f1']:.4f}"
        )
        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            torch.save(model.state_dict(), out_dir / "best.pt")
    model.load_state_dict(load_model_state(out_dir / "best.pt", device))
    test_metrics = evaluate(model, test_loader, device, id2label)
    result = {
        "seed": args.seed,
        "lambda_paper": args.lambda_paper,
        "grouping_source_train": args.grouping_source_train,
        "best_dev_f1": best_dev_f1,
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_macro_f1": test_metrics["macro_f1"],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "test_report.txt").write_text(test_metrics["report_text"], encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result
def test_only(paths, args):
    seed_all(args.seed)
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.checkpoint is None:
        raise ValueError(
            "Test mode requires --checkpoint. Example: "
            "--checkpoint outputs/run_seed42/best.pt"
        )
    checkpoint = Path(args.checkpoint)
    label2id, id2label = build_label_vocab()
    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_name,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    test_ds = Disease7000RefinedDataset(
        paths["test_file"],
        paths["feature_root"] / "test",
        tokenizer,
        label2id,
        args.max_seq,
    )
    if not args.no_strict_counts and len(test_ds) != EXPECTED_COUNTS["test"]:
        raise RuntimeError(
            f"test: {len(test_ds)} samples, "
            f"expected {EXPECTED_COUNTS['test']}"
        )
    collate = lambda batch: collate_batch(
        batch,
        tokenizer.pad_token_id or 0,
        label2id["O"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model = build_model_for_dataset(
        test_ds,
        label2id,
        id2label,
        args,
        device,
    )
    print(f"Loading checkpoint: {checkpoint}")
    state_dict = load_model_state(checkpoint, device)
    model.load_state_dict(state_dict)
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        id2label,
    )
    result = {
        "checkpoint": str(checkpoint),
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_macro_f1": test_metrics["macro_f1"],
    }
    print("\n=== Test Results ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n=== Classification Report ===")
    print(test_metrics["report_text"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_only_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "test_only_report.txt").write_text(
        test_metrics["report_text"],
        encoding="utf-8",
    )

    print(f"\nSaved test-only results to: {out_dir}")
    return result
def run_reviewer_grid(paths, args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for grouping in args.grid_grouping:
        for lam in args.grid_lambdas:
            for seed in args.grid_seeds:
                run_dir = root / f"group-{grouping}_lambda-{lam}_seed-{seed}"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--mode", "train",
                    "--data-dir", str(args.data_dir),
                    "--image-dir", str(args.image_dir),
                    "--aux-root", str(args.aux_root),
                    "--feature-root", str(args.feature_root),
                    "--bert-name", args.bert_name,
                    "--vit-name", args.vit_name,
                    "--lambda-paper", str(lam),
                    "--grouping-source-train", grouping,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--dropout", str(args.dropout),
                    "--max-seq", str(args.max_seq),
                    "--align-dim", str(args.align_dim),
                    "--lstm-hidden", str(args.lstm_hidden),
                    "--lstm-layers", str(args.lstm_layers),
                    "--grad-clip", str(args.grad_clip),
                    "--out", str(run_dir),
                ]
                if args.device:
                    cmd.extend(["--device", args.device])
                if args.local_files_only:
                    cmd.append("--local-files-only")
                if args.no_strict_counts:
                    cmd.append("--no-strict-counts")
                # Propagate non-default file overrides if supplied.
                for flag, value in [
                    ("--train-file", args.train_file),
                    ("--dev-file", args.dev_file),
                    ("--test-file", args.test_file),
                    ("--train-aux-dict", args.train_aux_dict),
                    ("--dev-aux-dict", args.dev_aux_dict),
                    ("--test-aux-dict", args.test_aux_dict),
                    ("--train-crop-dir", args.train_crop_dir),
                    ("--dev-crop-dir", args.dev_crop_dir),
                    ("--test-crop-dir", args.test_crop_dir),
                ]:
                    if value:
                        cmd.extend([flag, str(value)])
                print("RUN", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)
                runs.append(json.loads((run_dir / "result.json").read_text(encoding="utf-8")))
    grouped = {}
    for r in runs:
        grouped.setdefault((r["grouping_source_train"], r["lambda_paper"]), []).append(r)
    ranking = []
    for (grouping, lam), values in grouped.items():
        ranking.append(
            {
                "grouping": grouping,
                "lambda": lam,
                "mean_dev_f1": float(np.mean([x["best_dev_f1"] for x in values])),
            }
        )
    ranking.sort(key=lambda x: x["mean_dev_f1"], reverse=True)
    best = ranking[0]
    best_runs = grouped[(best["grouping"], best["lambda"])]
    def mean_std(key):
        values = [x[key] for x in best_runs]
        return float(np.mean(values)), float(np.std(values, ddof=1))
    p_mean, p_std = mean_std("test_precision")
    r_mean, r_std = mean_std("test_recall")
    f_mean, f_std = mean_std("test_f1")
    mf_mean, mf_std = mean_std("test_macro_f1")
    summary = {
        "label": "AgriFuseNER(approximate reimplementation)",
        "selection_rule": "highest mean validation F1; test set never used for hyperparameter selection",
        "ranking": ranking,
        "selected": best,
        "test_precision_mean": p_mean,
        "test_precision_std": p_std,
        "test_recall_mean": r_mean,
        "test_recall_std": r_std,
        "test_f1_mean": f_mean,
        "test_f1_std": f_std,
        "test_macro_f1_mean": mf_mean,
        "test_macro_f1_std": mf_std,
    }
    (root / "reviewer_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
def build_parser():
    p = argparse.ArgumentParser(
        description="AgriFuseNER approximate reimplementation on Disease7000-Refined"
    )
    p.add_argument("--mode", choices=["validate", "extract", "train", "test", "reviewer_grid", "all"], default="train")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--image-dir", default=str(IMAGE_DIR))
    p.add_argument("--aux-root", default=str(AUX_ROOT))
    p.add_argument("--feature-root", default="data/agrifusener_vit_features")
    p.add_argument("--train-file")
    p.add_argument("--dev-file")
    p.add_argument("--test-file")
    p.add_argument("--train-aux-dict")
    p.add_argument("--dev-aux-dict")
    p.add_argument("--test-aux-dict")
    p.add_argument("--train-crop-dir")
    p.add_argument("--dev-crop-dir")
    p.add_argument("--test-crop-dir")
    p.add_argument("--no-strict-counts", action="store_true")
    p.add_argument(
        "--bert-name",
        default=BERT_MODEL_NAME,
        help="Pretrained BERT model name or local path"
    )

    p.add_argument(
        "--vit-name",
        default=VIT_MODEL_NAME,
        help="Pretrained ViT model name or local path"
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached or downloaded pretrained models. "
             "By default, missing pretrained models are automatically downloaded."
    )
    p.add_argument("--max-seq", type=int, default=256)
    p.add_argument("--top-k", type=int, default=3, help="same ROI proposal count already used by AgriAlignNER")
    p.add_argument("--align-dim", type=int, default=768)
    p.add_argument("--lstm-hidden", type=int, default=384)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--grouping-source-train", choices=["predicted", "gold"], default="predicted")
    p.add_argument("--lambda-paper", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=PAPER_EPOCHS)
    p.add_argument("--batch-size", type=int, default=PAPER_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=PAPER_LR)
    p.add_argument("--dropout", type=float, default=PAPER_DROPOUT)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs/agrifusener")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint for --mode test, e.g. outputs/run_seed2021/best.pt",
    )
    p.add_argument("--grid-lambdas", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.5])
    p.add_argument("--grid-seeds", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    p.add_argument(
        "--grid-grouping",
        nargs="+",
        choices=["predicted", "gold"],
        default=["predicted"]
    )
    return p
def main():
    args = build_parser().parse_args()
    paths = resolve_paths(args)
    if args.mode == "validate":
        validate_data(paths, strict_counts=not args.no_strict_counts)
    elif args.mode == "extract":
        validate_data(paths, strict_counts=not args.no_strict_counts)
        extract_all(paths, args)
    elif args.mode == "train":
        train_one(paths, args)
    elif args.mode == "test":
        test_only(paths, args)
    elif args.mode == "reviewer_grid":
        run_reviewer_grid(paths, args)
    elif args.mode == "all":
        validate_data(paths, strict_counts=not args.no_strict_counts)
        extract_all(paths, args)
        run_reviewer_grid(paths, args)
if __name__ == "__main__":
    main()
