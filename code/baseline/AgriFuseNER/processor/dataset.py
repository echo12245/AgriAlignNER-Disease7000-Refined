from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset
ENTITY_TYPES = ["Crop", "Disease", "Feature", "Position"]
def build_label_vocab(entity_types=ENTITY_TYPES):
    labels = ["O"]
    for t in entity_types:
        labels.extend([f"B-{t}", f"I-{t}"])
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label
def parse_imgid_bio_file(path):
    path = Path(path)
    samples, imgid, tokens, labels = [], None, [], []
    def flush():
        nonlocal imgid, tokens, labels
        if imgid is None and not tokens:
            return
        if imgid is None or len(tokens) != len(labels):
            raise ValueError(
                f"Malformed block in {path}: IMGID={imgid}, "
                f"tokens={len(tokens)}, labels={len(labels)}"
            )
        samples.append({"id": imgid, "tokens": tokens, "labels": labels})
        imgid, tokens, labels = None, [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip("\ufeff")
        if line.startswith("IMGID:"):
            if tokens:
                flush()
            imgid = line.split("IMGID:", 1)[1].strip()
        elif not line.strip():
            if tokens:
                flush()
        else:
            parts = line.rstrip().split("\t")
            if len(parts) < 2:
                raise ValueError(f"Expected token<TAB>BIO label: {raw!r}")
            tokens.append(parts[0])
            labels.append(parts[1].strip())
    if tokens:
        flush()
    return samples
def load_visual_feature(path: Path):
    obj: Any = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        feat = obj.float()
        return feat, torch.ones(feat.shape[0], dtype=torch.bool)
    feat = obj.get("features", obj.get("visual_features"))
    mask = obj.get("mask", obj.get("visual_mask"))
    if feat is None:
        raise KeyError(f"No visual feature tensor in {path}")
    if mask is None:
        mask = torch.ones(feat.shape[0], dtype=torch.bool)
    return feat.float(), mask.bool()
class Disease7000RefinedDataset(Dataset):
    def __init__(self, text_file, feature_dir, tokenizer, label2id, max_length=256):
        self.samples = parse_imgid_bio_file(text_file)
        self.feature_dir = Path(feature_dir)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = int(max_length)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        row = self.samples[idx]
        enc = self.tokenizer(
            row["tokens"],
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = enc.word_ids()
        if word_ids is None:
            raise RuntimeError("A fast tokenizer is required (word_ids() unavailable).")
        label_ids, word_start_mask = [], []
        previous_word = None
        for wid in word_ids:
            if wid is None:
                continue
            label = row["labels"][wid]
            # Extra subtokens after the first inherit I-type rather than B-type.
            if wid == previous_word and label.startswith("B-"):
                label = "I-" + label[2:]
            if label not in self.label2id:
                raise KeyError(f"Unknown BIO label {label!r} in sample {row['id']}")
            label_ids.append(self.label2id[label])
            word_start_mask.append(wid != previous_word)
            previous_word = wid
        feature_path = self.feature_dir / f"{Path(row['id']).stem}.pt"
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Missing visual feature: {feature_path}\n"
                "Run --mode extract first."
            )
        visual_features, visual_mask = load_visual_feature(feature_path)
        return {
            "id": row["id"],
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.ones(len(enc["input_ids"]), dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "word_start_mask": torch.tensor(word_start_mask, dtype=torch.bool),
            "visual_features": visual_features,
            "visual_mask": visual_mask,
        }
def collate_batch(batch, pad_token_id=0, label_pad_id=0):
    batch_size = len(batch)
    max_len = max(len(x["input_ids"]) for x in batch)
    max_regions = max(x["visual_features"].shape[0] for x in batch)
    visual_dim = batch[0]["visual_features"].shape[-1]
    ids = []
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), label_pad_id, dtype=torch.long)
    word_start_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    visual_features = torch.zeros((batch_size, max_regions, visual_dim), dtype=torch.float)
    visual_mask = torch.zeros((batch_size, max_regions), dtype=torch.bool)
    for b, item in enumerate(batch):
        n = len(item["input_ids"])
        m = item["visual_features"].shape[0]
        input_ids[b, :n] = item["input_ids"]
        attention_mask[b, :n] = 1
        labels[b, :n] = item["labels"]
        word_start_mask[b, :n] = item["word_start_mask"]
        visual_features[b, :m] = item["visual_features"]
        visual_mask[b, :m] = item["visual_mask"]
        ids.append(item["id"])
    return {
        "ids": ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "word_start_mask": word_start_mask,
        "visual_features": visual_features,
        "visual_mask": visual_mask,
    }
