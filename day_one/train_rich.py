"""Train the Day-1 model on the RICHER concept set (60 concepts incl. confusable
near-duplicate colors) for Day-4.5 confirmation. Saves to out/worldfield_rich.pt.
Identical pipeline to train.py — only the config changes."""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config_rich import RichConfig
from data import ShapesDataset
from model import Worldfield, info_nce
from train import pick_device, retrieval_metrics

OUT = os.path.join(os.path.dirname(__file__), "out")


def main():
    cfg = RichConfig()
    torch.manual_seed(cfg.seed)
    os.makedirs(OUT, exist_ok=True)
    dev = pick_device()
    print(f"[rich] device {dev} | {cfg.num_classes} concepts | "
          f"chance R@1 {1/cfg.num_classes:.3f}")

    train_ds = ShapesDataset(cfg, "train")
    val_ds = ShapesDataset(cfg, "val")
    tdl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    vdl = DataLoader(val_ds, batch_size=256, shuffle=False)
    model = Worldfield(cfg, train_ds.vocab.size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for img, ids, _ in tdl:
            img, ids = img.to(dev), ids.to(dev)
            zi, zt, recon = model(img, ids)
            loss = info_nce(zi, zt, cfg.temperature) + cfg.recon_weight * F.mse_loss(recon, img)
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 7 == 0 or epoch == cfg.epochs:
            r_it, r_ti, *_ = retrieval_metrics(model, vdl, dev)
            print(f"[rich] epoch {epoch:3d} | R@1 i→t {r_it:.3f} | t→i {r_ti:.3f}")

    names = [f"{c} {s}" for (c, s) in train_ds.classes]
    torch.save({"model": model.state_dict(), "vocab_size": train_ds.vocab.size,
                "class_names": names}, os.path.join(OUT, "worldfield_rich.pt"))
    print(f"[rich] saved {OUT}/worldfield_rich.pt ({len(names)} concepts)")


if __name__ == "__main__":
    main()
