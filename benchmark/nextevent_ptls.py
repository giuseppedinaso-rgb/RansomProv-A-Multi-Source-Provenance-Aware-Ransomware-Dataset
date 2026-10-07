#!/usr/bin/env python3
"""Baseline multi-attributo con pytorch-lifestream (ptls), protocollo nextevent_data.

Architettura (come GptPretrainModule di ptls): TrxEncoder (embedding di ogni attributo dell'evento + intervallo
continuo log1p) -> encoder di sequenza causale -> una testa per attributo che
prevede l'evento SUCCESSIVO.
  GRU          ptls RnnEncoder
  Transformer  ptls GptEncoder (GPT-2 causale)
Addestramento: finestre di 128 eventi per gang (passo 128), AdamW, early
stopping sulla loss di validazione (pazienza 5), poi metriche sul test.

Uso: python nextevent_ptls.py <gru|transformer|tabformer|flat|mamba> <observed|enriched> [seed]
"""
import sys, json, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn import TrxEncoder
from ptls.nn.seq_encoder import RnnEncoder, GptEncoder
from nextevent_data import load, targets, metrics, FIELDS, IGNORE, ROOT

OUT = ROOT / "benchmark" / "results" / "nextevent"
WIN, BATCH, MAX_EPOCHS, PATIENCE = 128, 32, 60, 5


CAT = FIELDS + ["gang"]


class FieldTransformer(nn.Module):
    """Encoder intra-evento in stile TabFormer (Padhi et al., ICASSP 2021):
    ogni attributo e' un token (con embedding di posizione del campo), un piccolo
    Transformer li combina, e la media dei token da' il vettore dell'evento."""
    def __init__(self, sizes, d):
        super().__init__()
        self.emb = nn.ModuleDict({f: nn.Embedding(n, d) for f, n in sizes.items()})
        self.gap = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.zeros(len(sizes) + 1, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True), 2)

    def forward(self, pay):
        toks = [self.emb[f](pay[f]) for f in self.emb] + [self.gap(pay["gap"].unsqueeze(-1))]
        x = torch.stack(toks, dim=2) + self.pos                       # (B, L, F, d)
        B, L, F, d = x.shape
        return self.enc(x.reshape(B * L, F, d)).mean(1).reshape(B, L, d)


class Model(nn.Module):
    """kind:
      gru          ptls TrxEncoder -> ptls RnnEncoder (GRU)
      transformer  ptls TrxEncoder -> ptls GptEncoder (GPT-2 causale)
      tabformer    encoder intra-evento (FieldTransformer) -> GptEncoder
      flat         token appiattiti (Stein et al., arXiv:2410.10648): ogni attributo
                   e' un token nella sequenza, GPT-2 causale; si legge lo stato
                   all'ultimo token di ogni evento
      mamba        ptls TrxEncoder -> Mamba (mambapy, implementazione PyTorch di
                   Gu & Dao 2023)
    In tutti i casi: una testa per attributo che prevede l'evento successivo."""
    def __init__(self, vocab, n_gang, kind, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.kind = kind
        sizes = {f: len(vocab[f]) + 1 for f in FIELDS}; sizes["gang"] = n_gang + 1
        if kind in ("gru", "transformer", "mamba"):
            emb = {f: {"in": sizes[f], "out": 16} for f in FIELDS}
            emb["gang"] = {"in": sizes["gang"], "out": 32}
            self.trx = TrxEncoder(embeddings=emb, numeric_values={"gap": "identity"}, use_batch_norm=False)
            self.proj = nn.Linear(self.trx.output_size, hidden)
        if kind == "gru":
            self.seq = RnnEncoder(input_size=hidden, hidden_size=hidden, type="gru", num_layers=layers, dropout=dropout)
        elif kind in ("transformer", "tabformer"):
            self.seq = GptEncoder(n_embd=hidden, n_layer=layers, n_head=4, n_inner=4 * hidden, n_positions=WIN + 8,
                                  use_start_random_shift=False, resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout)
            if kind == "tabformer":
                self.fields = FieldTransformer(sizes, hidden)
        elif kind == "flat":
            self.tok = nn.ModuleDict({f: nn.Embedding(n, hidden) for f, n in sizes.items()})
            self.gap = nn.Linear(1, hidden)
            self.ftype = nn.Embedding(len(sizes) + 1, hidden)
            self.seq = GptEncoder(n_embd=hidden, n_layer=layers, n_head=4, n_inner=4 * hidden,
                                  n_positions=WIN * (len(sizes) + 1) + 8, use_start_random_shift=False,
                                  resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout)
        elif kind == "mamba":
            from mambapy.mamba import Mamba, MambaConfig
            self.seq = Mamba(MambaConfig(d_model=hidden, n_layers=layers, d_state=16))
        self.heads = nn.ModuleDict({f: nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, len(vocab[f])))
                                    for f in FIELDS})

    def forward(self, pb):
        pay, lens = pb.payload, pb.seq_lens
        if self.kind == "flat":
            parts = [self.tok[f](pay[f]) + self.ftype.weight[i] for i, f in enumerate(self.tok)]
            parts.append(self.gap(pay["gap"].unsqueeze(-1)) + self.ftype.weight[len(self.tok)])
            x = torch.stack(parts, dim=2)                             # (B, L, F, d)
            B, L, F, d = x.shape
            h = self.seq(PaddedBatch(x.reshape(B, L * F, d), lens * F)).payload
            h = h.reshape(B, L, F, d)[:, :, -1]                       # ultimo token di ogni evento
        elif self.kind == "tabformer":
            h = self.seq(PaddedBatch(self.fields(pay), lens)).payload
        else:
            z = PaddedBatch(self.proj(self.trx(pb).payload), lens)
            h = self.seq(z).payload if self.kind != "mamba" else self.seq(z.payload)
        return {f: self.heads[f](h) for f in FIELDS}


def windows(ev, tgt):
    """Finestre per gang. Input = eventi [s, e); bersaglio in posizione i = evento i+1."""
    W = []
    for _, g in ev.groupby("gang", sort=False):
        idx = g.index.values
        for s in range(0, len(idx) - 1, WIN):
            inp = idx[s:min(s + WIN, len(idx) - 1)]
            nxt = inp + 1
            W.append((inp, nxt))
    return W


def to_batch(ev, tgt, items, split, dev):
    L = max(len(i) for i, _ in items)
    pay = {k: torch.zeros(len(items), L, dtype=torch.long) for k in FIELDS + ["gang"]}
    pay["gap"] = torch.zeros(len(items), L)
    ys = {f: torch.full((len(items), L), IGNORE, dtype=torch.long) for f in FIELDS}
    lens = []
    for b, (inp, nxt) in enumerate(items):
        n = len(inp); lens.append(n)
        pay["delta"][b, :n] = torch.tensor(ev.delta.values[inp]) + 1
        pay["country"][b, :n] = torch.tensor(ev.country_i.values[inp]) + 1
        pay["sector"][b, :n] = torch.tensor(ev.sector_i.values[inp]) + 1
        pay["gang"][b, :n] = torch.tensor(ev.gang_i.values[inp])
        pay["gap"][b, :n] = torch.tensor(ev.gap.values[inp], dtype=torch.float)
        keep = ev.split.values[nxt] == split
        for f in FIELDS:
            y = tgt[f][nxt].copy(); y[~keep] = IGNORE
            ys[f][b, :n] = torch.tensor(y)
    pb = PaddedBatch({k: v.to(dev) for k, v in pay.items()}, torch.tensor(lens).to(dev))
    return pb, {f: v.to(dev) for f, v in ys.items()}


def run_epoch(model, ev, tgt, W, split, dev, opt=None, train_loss=None):
    """train_loss: dict attributo -> nn.CrossEntropyLoss usata per l'addestramento
    (varianti: pesata per classe, label smoothing). La loss riportata e' sempre la
    cross-entropy semplice, cosi' la selezione sulla validazione e' confrontabile."""
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")
    tot, n = 0.0, 0
    probs = {f: [] for f in FIELDS}; trues = {f: [] for f in FIELDS}
    order = list(range(len(W)))
    if opt: random.shuffle(order)
    for i in range(0, len(order), BATCH):
        pb, ys = to_batch(ev, tgt, [W[j] for j in order[i:i + BATCH]], split, dev)
        with torch.set_grad_enabled(opt is not None):
            out = model(pb)
            flat = {f: (out[f].reshape(-1, out[f].shape[-1]), ys[f].reshape(-1)) for f in FIELDS}
            loss = sum(ce(*flat[f]) for f in FIELDS)
            cnt = sum(int((ys[f] != IGNORE).sum()) for f in FIELDS)
        if opt and cnt:
            tl = loss if train_loss is None else sum(train_loss[f](*flat[f]) for f in FIELDS)
            opt.zero_grad(); (tl / cnt).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += loss.item(); n += cnt
        if opt is None:
            for f in FIELDS:
                lg, y = flat[f]; keep = y != IGNORE
                probs[f].append(lg[keep].float().softmax(-1).cpu().numpy()); trues[f].append(y[keep].cpu().numpy())
    res = {"loss": tot / max(n, 1)}
    if opt is None:
        for f in FIELDS:
            P, y = np.concatenate(probs[f]), np.concatenate(trues[f])
            # classi che non sono mai un bersaglio: 'unknown' (paese, settore), 'first' (intervallo)
            P[:, -1 if f != "delta" else 0] = 0.0
            P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
            res[f] = metrics(y, P.argmax(1), P)
    return res


DEFAULTS = {"lr": 1e-3, "hidden": 128, "layers": 2, "dropout": 0.1, "weight_decay": 0.01, "loss": "ce"}


def train_eval(kind, version, seed, params=None, save=True, tag="", source=None, eval_ids=None):
    """eval_ids: se indicato, il TEST si valuta solo su questi event_id (stessi
    eventi per tutte le viste a fonte singola e per il dataset unito)."""
    """Addestra con early stopping sulla validazione; restituisce i risultati."""
    global WIN
    p = {**DEFAULTS, **(params or {})}
    WIN = int(p.get("win", WIN))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev, vocab = load(version, source); tgt = targets(ev, vocab); W = windows(ev, tgt)
    if eval_ids is not None:
        out_of_set = (ev.split.values == "test") & ~ev.event_id.isin(eval_ids).values
        for f in FIELDS:
            tgt[f][out_of_set] = IGNORE
    model = Model(vocab, int(ev.gang_i.max()), kind, p["hidden"], p["layers"], p["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    train_loss = None
    if p["loss"] == "smooth":
        train_loss = {f: nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum", label_smoothing=0.1) for f in FIELDS}
    elif p["loss"] == "weighted":
        train_loss = {}
        tr = ev.split.values == "train"
        for f in FIELDS:
            y = tgt[f][tr]; y = y[y != IGNORE]
            cnt = np.bincount(y, minlength=len(vocab[f])).astype(float)
            w = np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0.0); w = w / w[cnt > 0].mean()
            train_loss[f] = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float, device=dev),
                                                ignore_index=IGNORE, reduction="sum")
    best, bad, state = float("inf"), 0, None
    for ep in range(MAX_EPOCHS):
        model.train(); tr_ = run_epoch(model, ev, tgt, W, "train", dev, opt, train_loss)
        model.eval(); va = run_epoch(model, ev, tgt, W, "val", dev)
        if va["loss"] < best - 1e-4:
            best, bad, state = va["loss"], 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state); model.eval()
    te = run_epoch(model, ev, tgt, W, "test", dev)
    va = run_epoch(model, ev, tgt, W, "val", dev)
    res = {"model": {"gru": "ptls-gru", "transformer": "ptls-transformer"}.get(kind, kind), "version": version,
           "seed": seed, "source": source or "merged", "params": p, "best_val_loss": best, "val": va, "test": te}
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(OUT / f"{res['model']}{tag}_{version}_s{seed}.json", "w"), indent=2)
    return res


def main():
    kind, version = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    res = train_eval(kind, version, seed)
    print(json.dumps({f: round(res["test"][f]["weighted_f1"], 4) for f in FIELDS}))


if __name__ == "__main__":
    main()
