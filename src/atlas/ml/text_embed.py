"""Neural news model: transformer embeddings + a trained neural network head.

Following Chen, Kelly & Xiu ("Expected Returns and Large Language Models"): a pretrained
language model turns each article into a contextual vector; a predictor trained on our
own data maps vectors to expected market reactions.

Contamination control: the encoder is pinned to a revision published before the test
period (all-mpnet-base-v2 @ 2024-11-05, trained on 2021-era data, Apache-2.0), so it
cannot carry memorised knowledge of 2025-2026 events. Company-identity words are removed
before encoding so the head learns what news says, not whom it is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ENCODER = "sentence-transformers/all-mpnet-base-v2"
ENCODER_REVISION = "9a3225965996d404b775526de6dbfe85d3368642"  # 2024-11-05


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder:
    def __init__(
        self,
        name: str = ENCODER,
        revision: str = ENCODER_REVISION,
        max_length: int = 128,
        batch_size: int = 128,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(name, revision=revision).eval().to(device())
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        """Mean-pooled, L2-normalised sentence embeddings (float16 to save space)."""
        out = []
        dev = device()
        for i in range(0, len(texts), self.batch_size):
            batch = self.tok(
                texts[i : i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(dev)
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = nn.functional.normalize(emb, dim=-1)
            out.append(emb.cpu().numpy().astype(np.float16))
        return np.concatenate(out) if out else np.zeros((0, 768), np.float16)


def mask_text(text: pd.Series, identity: set[str]) -> pd.Series:
    return text.map(lambda t: " ".join(w for w in t.split() if w not in identity))


def embed_cached(
    rows: pd.DataFrame,
    identity: set[str],
    cache: str | Path,
    encoder: Encoder | None = None,
    progress=print,
) -> np.ndarray:
    """Embeddings for rows (one per article id; text masked), cached on disk by article id."""
    cache = Path(cache)
    arts = rows.drop_duplicates("id")[["id", "text"]]
    known: dict[int, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache)
        known = dict(zip(z["ids"].tolist(), z["emb"], strict=True))
    todo = arts[~arts["id"].isin(known)]
    if len(todo):
        enc = encoder or Encoder()
        texts = mask_text(todo["text"], identity).tolist()
        embs = []
        step = 20_000
        for i in range(0, len(texts), step):
            embs.append(enc.encode(texts[i : i + step]))
            progress(f"  embedded {min(i + step, len(texts)):,}/{len(texts):,} articles")
        new = np.concatenate(embs)
        known.update(zip(todo["id"].tolist(), new, strict=True))
        cache.parent.mkdir(parents=True, exist_ok=True)
        ids = np.array(list(known))
        np.savez(cache, ids=ids, emb=np.stack([known[i] for i in ids]))
    return np.stack([known[i] for i in rows["id"].tolist()]).astype(np.float32)


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class NeuralHead:
    """Binary classifier: clear positive vs clear negative reaction. Early stopping on a
    LATER time block than training (never a random split of the same period)."""

    hidden: int = 256
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 30
    patience: int = 4
    batch: int = 1024
    label_quantile: float = 0.3
    seed: int = 0
    _model: _MLP | None = field(default=None, repr=False)
    history: list[tuple[int, float]] = field(default_factory=list)

    @staticmethod
    def _labels(reaction: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = np.nanquantile(reaction, q), np.nanquantile(reaction, 1 - q)
        keep = (reaction <= lo) | (reaction >= hi)
        return keep, (reaction >= hi).astype(np.float32)

    def fit(
        self, x_tr: np.ndarray, r_tr: np.ndarray, x_va: np.ndarray, r_va: np.ndarray
    ) -> NeuralHead:
        torch.manual_seed(self.seed)
        keep, y = self._labels(r_tr, self.label_quantile)
        lo, hi = (
            np.nanquantile(r_tr, self.label_quantile),
            np.nanquantile(r_tr, 1 - self.label_quantile),
        )
        vkeep = (r_va <= lo) | (r_va >= hi)
        vy = (r_va >= hi).astype(np.float32)
        dev = device()
        xt = torch.tensor(x_tr[keep], device=dev)
        yt = torch.tensor(y[keep], device=dev)
        xv = torch.tensor(x_va[vkeep], device=dev)
        yv = torch.tensor(vy[vkeep], device=dev)
        model = _MLP(xt.shape[1], self.hidden, self.dropout).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(len(xt), device=dev)
            for i in range(0, len(xt), self.batch):
                idx = perm[i : i + self.batch]
                opt.zero_grad()
                loss_fn(model(xt[idx]), yt[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vloss = float(loss_fn(model(xv), yv))
            self.history.append((epoch, vloss))
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model.eval()
        return self

    @torch.no_grad()
    def score(self, x: np.ndarray) -> np.ndarray:
        """Probability of a positive reaction minus 0.5."""
        assert self._model is not None
        out = []
        dev = device()
        for i in range(0, len(x), 8192):
            logits = self._model(torch.tensor(x[i : i + 8192], device=dev))
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out) - 0.5
