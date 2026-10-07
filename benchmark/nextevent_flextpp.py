#!/usr/bin/env python3
"""FlexTPP (Draxler et al., NeurIPS 2025) sul protocollo nextevent_data.

Implementazione: https://github.com/czi-ai/FlexTPP (non ufficiale, MIT), usata
senza modifiche dalla cartella <FLEXTPP_DIR>; ambiente Python 3.12
(ambiente separato).

FlexTPP rappresenta ogni evento come una sequenza di valori a tipo misto e
modella ciascun valore condizionato su tutti i precedenti, inclusi quelli
dello STESSO evento. Per prevedere un attributo usando solo la storia (come
fanno gli altri modelli del protocollo), l'attributo deve essere il primo
dell'evento. Si addestrano quindi tre ordini:
  CKS  paese, settore, tempo   -> si valuta il paese
  KCS  settore, paese, tempo   -> si valuta il settore
  SCK  tempo, paese, settore   -> si valuta l'intervallo (mediana prevista ->
                                  stesso giorno / entro 7 giorni / oltre)
Tempo = giorni dall'evento precedente della gang (continuo, trasformazione
logaritmica come nel codice originale). Paese e settore condividono la tabella
di embedding di FlexTPP: gli indici del settore sono spostati dopo quelli del
paese; la predizione e' ristretta alle classi dell'attributo, senza 'unknown'.

Addestramento: finestre di 128 eventi per gang dentro ciascun periodo
(training, validazione); AdamW 8e-4 con OneCycle come nella configurazione
originale; early stopping sulla NLL di validazione (pazienza 10, max 100
epoche); modello model-100k (depth 2, dim_k 16, 4 teste, dim_ff 256) con
monotonic_bins=11 come nelle configurazioni EasyTPP del repository (128, il
default, non e' compatibile con FrEIA).
Valutazione: finestre su tutta la storia, si leggono i parametri della prima
posizione di ogni evento di test.

Uso: PYTHONPATH=<FLEXTPP_DIR> python nextevent_flextpp.py <observed|enriched> [seed]
"""
import sys, json, math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf
from flex_tpp.model_pl import FlexTPPModule
from flex_tpp.data.base import ItemSpec, batch_collate, MODALITY_CONTINUOUS, MODALITY_CATEGORICAL, log_and_log_abs_det, reverse_log
from nextevent_data import load, targets, metrics, IGNORE, ROOT

OUT = ROOT / "benchmark" / "results" / "nextevent"
WIN = 128
ORDERS = {"CKS": "country", "KCS": "sector", "SCK": "delta"}


def item(ev, idx, order, n_c):
    data, types, corr = [], [], []
    for i in idx:
        for e in order:
            if e == "S":
                v, c = log_and_log_abs_det(float(ev.delta_days.values[i]) if not math.isnan(ev.delta_days.values[i]) else 0.0)
                data.append(v); types.append(MODALITY_CONTINUOUS); corr.append(c)
            elif e == "C":
                data.append(float(ev.country_i.values[i])); types.append(MODALITY_CATEGORICAL); corr.append(0.0)
            else:
                data.append(float(ev.sector_i.values[i] + n_c)); types.append(MODALITY_CATEGORICAL); corr.append(0.0)
    return ItemSpec(torch.tensor(data, dtype=torch.float32), torch.tensor(types), torch.tensor(corr).float(), None, {})


class Seqs(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def chunks(ev, split=None):
    out = []
    for _, g in ev.groupby("gang", sort=False):
        idx = g.index.values if split is None else g.index.values[g.split.values == split]
        out += [idx[s:s + WIN] for s in range(0, len(idx), WIN) if len(idx[s:s + WIN]) >= 2]
    return out


def train_order(ev, order, n_c, n_cls, seed, dev_id):
    pl.seed_everything(seed, workers=True)
    mk = lambda split: Seqs([item(ev, ix, order, n_c) for ix in chunks(ev, split)])
    tr, va = mk("train"), mk("val")
    kw = dict(batch_size=32, collate_fn=batch_collate, num_workers=0)
    optim = OmegaConf.create({"optimizer": {"_target_": "torch.optim.AdamW", "lr": 8e-4},
                              "lr_scheduler": {"scheduler": {"_target_": "torch.optim.lr_scheduler.OneCycleLR",
                                                             "total_steps": 100, "div_factor": 10, "pct_start": 0.1},
                                               "interval": "epoch", "frequency": 1}})
    model = FlexTPPModule(dims=[WIN * len(order)], dim_k=16, n_head=4, dim_ff=256, depth=2, non_linearity="GELU",
                          normalize=True, dropout=0.1, monotonic_bins=11, max_num_classes=n_cls, optim_cfg=optim)
    ck = ModelCheckpoint(monitor="val/nll", mode="min", save_top_k=1,
                         dirpath=str(OUT / "flextpp_ckpt" / f"{order}_s{seed}"))
    trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=[dev_id], logger=False, enable_progress_bar=False,
                         callbacks=[EarlyStopping(monitor="val/nll", patience=10, check_finite=False), ck],
                         gradient_clip_val=1.0, precision="bf16-mixed", enable_model_summary=False)
    trainer.fit(model, DataLoader(tr, shuffle=True, **kw), DataLoader(va, **kw))
    best = FlexTPPModule.load_from_checkpoint(ck.best_model_path, weights_only=False)
    return best.model.eval().cuda(), float(ck.best_model_score)


@torch.no_grad()
def predict_first(model, ev, order, n_c, n_cls, split):
    """Predizione dell'attributo in prima posizione per ogni evento del periodo `split`."""
    preds, rows = [], []
    rng = {"C": (0, n_c), "K": (n_c, n_cls)}
    for _, g in ev.groupby("gang", sort=False):
        idx = g.index.values
        for s in range(0, len(idx), WIN):
            ix = idx[s:s + WIN]
            keep = np.where(ev.split.values[ix] == split)[0]
            if not len(keep):
                continue
            b = batch_collate([item(ev, ix, order, n_c)])
            b = type(b)(b.data.cuda(), b.types.cuda(), b.log_prob_correction.cuda(), None, b.extras)
            _, params, masks = model.log_prob(b, return_parameters=True, return_masks=True)
            pos = keep * len(order)
            first = order[0]
            mod = MODALITY_CONTINUOUS if first == "S" else MODALITY_CATEGORICAL
            mask = masks[mod][0]
            where = torch.nonzero(mask, as_tuple=False)[:, 0].tolist()
            par = params[mod]
            lookup = {p: k for k, p in enumerate(where)}
            for p, r in zip(pos, ix[keep]):
                pp = par[lookup[int(p)]]
                if first == "S":
                    days = float(reverse_log(model.monotonic_network.sample(pp[None], median=True)[0]))
                    preds.append(1 if days < 0.5 else (2 if days <= 7 else 3))
                else:
                    lo, hi = rng[first]
                    logits = pp[lo:hi].float().clone(); logits[-1] = -1e9          # niente 'unknown'
                    preds.append(int(logits.argmax()))
                rows.append(r)
    return np.array(rows), np.array(preds)


def main():
    version = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    torch.set_num_threads(4)
    ev, vocab = load(version); tgt = targets(ev, vocab)
    n_c = len(vocab["country"]); n_cls = n_c + len(vocab["sector"])
    res = {"model": "FlexTPP", "version": version, "seed": seed, "val": {}, "test": {}, "best_val_nll": {}}
    for order, field in ORDERS.items():
        model, nll = train_order(ev, order, n_c, n_cls, seed, 0)
        res["best_val_nll"][order] = nll
        for split in ("val", "test"):
            rows, p = predict_first(model, ev, order, n_c, n_cls, split)
            res[split][field] = metrics(tgt[field][rows], p)
        print(order, field, round(res["test"][field]["weighted_f1"], 4), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"FlexTPP_{version}_s{seed}.json", "w"), indent=2)
    print(json.dumps({f: round(res["test"][f]["weighted_f1"], 4) for f in res["test"]}))


if __name__ == "__main__":
    main()
