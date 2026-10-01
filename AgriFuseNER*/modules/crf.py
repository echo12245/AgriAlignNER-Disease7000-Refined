from __future__ import annotations
import torch
from torch import nn
class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = int(num_tags)
        self.start = nn.Parameter(torch.empty(num_tags))
        self.end = nn.Parameter(torch.empty(num_tags))
        self.trans = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.uniform_(self.start, -0.1, 0.1)
        nn.init.uniform_(self.end, -0.1, 0.1)
        nn.init.uniform_(self.trans, -0.1, 0.1)
    def _score(self, emissions, tags, mask):
        B, L, _ = emissions.shape
        mask = mask.bool()
        score = self.start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, L):
            mt = mask[:, t].to(emissions.dtype)
            emit = emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)
            tr = self.trans[tags[:, t-1], tags[:, t]]
            score = score + mt * (emit + tr)
        last = mask.long().sum(1).clamp_min(1) - 1
        last_tags = tags.gather(1, last[:, None]).squeeze(1)
        return score + self.end[last_tags]
    def _log_partition(self, emissions, mask):
        B, L, _ = emissions.shape
        mask = mask.bool()
        alpha = self.start + emissions[:, 0]
        for t in range(1, L):
            nxt = torch.logsumexp(alpha[:, :, None] + self.trans[None, :, :], dim=1) + emissions[:, t]
            alpha = torch.where(mask[:, t:t+1], nxt, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)
    def forward(self, emissions, tags, mask, reduction="mean"):
        llh = self._score(emissions, tags, mask) - self._log_partition(emissions, mask)
        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        return llh.mean()
    @torch.no_grad()
    def decode(self, emissions, mask):
        B, L, C = emissions.shape
        mask = mask.bool()
        score = self.start + emissions[:, 0]
        history = []
        for t in range(1, L):
            cand = score[:, :, None] + self.trans[None, :, :]
            best_score, best_prev = cand.max(dim=1)
            best_score = best_score + emissions[:, t]
            score = torch.where(mask[:, t:t+1], best_score, score)
            history.append(best_prev)
        score = score + self.end
        best_last = score.argmax(1)
        lengths = mask.long().sum(1)
        paths = []
        for b in range(B):
            n = int(lengths[b])
            tag = int(best_last[b])
            path = [tag]
            for t in range(n - 1, 0, -1):
                tag = int(history[t - 1][b, tag])
                path.append(tag)
            paths.append(list(reversed(path)))
        return paths
