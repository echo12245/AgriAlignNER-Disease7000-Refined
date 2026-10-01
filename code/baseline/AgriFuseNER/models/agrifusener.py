from __future__ import annotations
import math
import torch
from torch import nn
from ..modules.crf import LinearChainCRF
B_ID = 0
M_ID = 1
def bio_ids_to_bm(label_ids: torch.Tensor, id2label: dict[int, str]) -> torch.Tensor:
    out = torch.full_like(label_ids, B_ID)
    for idx, name in id2label.items():
        if name.startswith("I-"):
            out = torch.where(label_ids == idx, torch.full_like(out, M_ID), out)
    return out
def boundary_to_group_ids(boundary_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, L = boundary_ids.shape
    groups = torch.zeros_like(boundary_ids)
    for b in range(B):
        gid = 0
        for t in range(L):
            if not bool(mask[b, t]):
                continue
            if gid == 0 or int(boundary_ids[b, t]) == B_ID:
                gid += 1
            groups[b, t] = gid
    return groups
def same_group_average(token_features: torch.Tensor, group_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, _, _ = token_features.shape
    out = torch.zeros_like(token_features)
    for b in range(B):
        valid = mask[b].bool()
        g = group_ids[b]
        same = (
            (g[:, None] == g[None, :])
            & valid[:, None]
            & valid[None, :]
            & (g[:, None] > 0)
        )
        W = same.to(token_features.dtype)
        W = W / W.sum(-1, keepdim=True).clamp_min(1.0)
        out[b] = W @ token_features[b]
        out[b, ~valid] = 0.0
    return out
class DCGA(nn.Module):
    def __init__(self, text_dim: int, visual_dim: int, align_dim: int = 768):
        super().__init__()
        self.align_dim = int(align_dim)
        self.q = nn.Linear(text_dim, align_dim, bias=False)
        self.k = nn.Linear(visual_dim, align_dim, bias=False)
        self.value = nn.Identity() if visual_dim == text_dim else nn.Linear(visual_dim, text_dim, bias=False)
        self.gate = nn.Linear(text_dim + 1, 1)
        self.norm = nn.LayerNorm(text_dim)
    def forward(self, F, V, visual_mask=None):
        S = (self.q(F) @ self.k(V).transpose(-1, -2)) / math.sqrt(self.align_dim)
        if visual_mask is not None:
            S = S.masked_fill(~visual_mask[:, None, :].bool(), -1e4)
        max_s = S.max(-1).values.unsqueeze(-1)
        gate = torch.sigmoid(self.gate(torch.cat([F, max_s], dim=-1)))
        attention = torch.softmax(S, dim=-1)
        if visual_mask is not None:
            attention = attention * visual_mask[:, None, :].to(attention.dtype)
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        evidence = attention @ self.value(V)
        fused = self.norm(F + gate * evidence)
        return fused, S, attention, gate
class AgriFuseNERApprox(nn.Module):
    def __init__(
        self,
        num_tags,
        id2label,
        text_encoder,
        visual_dim,
        align_dim=768,
        lstm_hidden=384,
        lstm_layers=1,
        dropout=0.1,
        lambda_paper=0.1,
        grouping_source_train="predicted",
        local_files_only=False,
    ):
        super().__init__()
        self.id2label = id2label
        self.lambda_paper = float(lambda_paper)
        self.grouping_source_train = grouping_source_train

        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(
            text_encoder, local_files_only=bool(local_files_only)
        )
        text_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.boundary_head = nn.Linear(text_dim, 2)
        self.boundary_crf = LinearChainCRF(2)
        self.entity_norm = nn.LayerNorm(text_dim)
        self.dcga = DCGA(text_dim, visual_dim, align_dim)
        self.bilstm = nn.LSTM(
            text_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.ner_head = nn.Linear(2 * lstm_hidden, num_tags)
        self.ner_crf = LinearChainCRF(num_tags)
    @staticmethod
    def _paths_to_tensor(paths, mask, device):
        out = torch.zeros(mask.shape, dtype=torch.long, device=device)
        for b, path in enumerate(paths):
            out[b, : len(path)] = torch.tensor(path, dtype=torch.long, device=device)
        return out
    def forward(self, input_ids, attention_mask, visual_features, visual_mask=None, labels=None):
        mask = attention_mask.bool()
        token_features = self.dropout(
            self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        )
        boundary_emissions = self.boundary_head(token_features)
        boundary_paths = self.boundary_crf.decode(boundary_emissions, mask)
        boundary_pred = self._paths_to_tensor(boundary_paths, mask, input_ids.device)
        boundary_gold = bio_ids_to_bm(labels, self.id2label) if labels is not None else None
        use_gold = self.training and self.grouping_source_train == "gold" and boundary_gold is not None
        boundary_for_grouping = boundary_gold if use_gold else boundary_pred
        groups = boundary_to_group_ids(boundary_for_grouping, mask)
        entity_features = self.entity_norm(same_group_average(token_features, groups, mask))
        fused, similarity, region_attention, gate = self.dcga(
            entity_features, visual_features, visual_mask
        )
        contextualized, _ = self.bilstm(fused)
        ner_emissions = self.ner_head(self.dropout(contextualized))
        ner_paths = self.ner_crf.decode(ner_emissions, mask)
        output = {
            "ner_paths": ner_paths,
            "boundary_paths": boundary_paths,
            "groups": groups,
            "similarity": similarity,
            "attention": region_attention,
            "gate": gate,
        }
        if labels is not None:
            ner_loss = -self.ner_crf(ner_emissions, labels, mask, reduction="mean")
            boundary_loss = -self.boundary_crf(
                boundary_emissions, boundary_gold, mask, reduction="mean"
            )
            total_loss = ner_loss + (1.0 - self.lambda_paper) * boundary_loss
            output.update(
                loss=total_loss,
                ner_loss=ner_loss,
                boundary_loss=boundary_loss,
            )
        return output
