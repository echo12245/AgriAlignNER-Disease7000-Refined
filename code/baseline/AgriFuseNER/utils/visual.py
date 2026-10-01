from __future__ import annotations
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from ..processor.dataset import parse_imgid_bio_file
def find_image(root, imgid):
    root = Path(root)
    stem = Path(imgid).stem
    candidates = [imgid, f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.png"]
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found for {imgid} under {root}")
def _encode(images, processor, vit, device):
    batch = processor(images=images, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = vit(**batch)
        features = (
            output.pooler_output
            if getattr(output, "pooler_output", None) is not None
            else output.last_hidden_state[:, 0]
        )
    return features.cpu()
def extract_split_features(
    text_file,
    image_dir,
    aux_dict,
    crop_dir,
    out_dir,
    vit_name,
    top_k=3,
    device=None,
    local_files_only=True,
):
    rows = parse_imgid_bio_file(text_file)
    mapping = torch.load(aux_dict, map_location="cpu")
    processor = AutoImageProcessor.from_pretrained(
        vit_name, local_files_only=bool(local_files_only)
    )
    vit = AutoModel.from_pretrained(
        vit_name, local_files_only=bool(local_files_only)
    )
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vit.to(device)
    crop_root = Path(crop_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(tqdm(rows, desc=f"ViT features: {Path(text_file).stem}")):
        global_image = Image.open(find_image(image_dir, row["id"])).convert("RGB")
        names = list(mapping.get(idx, []))[: int(top_k)]
        local_images, used_names = [], []
        for name in names:
            p = crop_root / name
            if p.exists():
                local_images.append(Image.open(p).convert("RGB"))
                used_names.append(str(name))
        features = _encode(local_images + [global_image], processor, vit, device)
        mask = torch.ones(features.shape[0], dtype=torch.bool)
        torch.save(
            {"features": features, "mask": mask, "roi_files": used_names},
            out_root / f"{Path(row['id']).stem}.pt",
        )
