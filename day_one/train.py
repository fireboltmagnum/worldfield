"""Train the day-one model and report the pass/fail signal: cross-modal
retrieval accuracy. Also dumps a latent-space plot and reconstructions.

This is the experiment that decides whether the core Worldfield claim is real
at small scale. Nothing here is meant to be the final architecture.
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from data import ShapesDataset
from model import Worldfield, info_nce
from viz import plot_latent_space, plot_reconstructions

OUT = os.path.join(os.path.dirname(__file__), "out")


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def retrieval_metrics(model, loader, device):
    """R@1 in both directions over the whole val set + a class-clustering check.

    Chance R@1 with N items is 1/N for unique-match retrieval, but since many
    items share a class we also report class-level R@1 (did the nearest item in
    the other modality have the same class?), whose chance is 1/num_classes.
    """
    model.eval()
    zis, zts, labels = [], [], []
    for img, ids, label in loader:
        img, ids = img.to(device), ids.to(device)
        zis.append(F.normalize(model.encode_image(img), dim=-1).cpu())
        zts.append(F.normalize(model.encode_text(ids), dim=-1).cpu())
        labels.append(label)
    zi = torch.cat(zis); zt = torch.cat(zts); lab = torch.cat(labels)

    sim = zi @ zt.t()  # [N_img, N_txt]
    # class-level R@1: nearest text to each image shares its class, and vice versa
    i2t = lab[sim.argmax(dim=1)]
    t2i = lab[sim.t().argmax(dim=1)]
    r1_i2t = (i2t == lab).float().mean().item()
    r1_t2i = (t2i == lab).float().mean().item()
    return r1_i2t, r1_t2i, zi, zt, lab


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    os.makedirs(OUT, exist_ok=True)
    device = pick_device()
    print(f"device: {device} | classes: {cfg.num_classes} | "
          f"chance R@1 ≈ {1/cfg.num_classes:.3f}")

    train_ds = ShapesDataset(cfg, "train")
    val_ds = ShapesDataset(cfg, "val")
    vocab_size = train_ds.vocab.size
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = Worldfield(cfg, vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"train: {len(train_ds)} val: {len(val_ds)}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot_c = tot_r = n = 0
        for img, ids, _ in train_dl:
            img, ids = img.to(device), ids.to(device)
            zi, zt, recon = model(img, ids)
            l_contrast = info_nce(zi, zt, cfg.temperature)
            l_recon = F.mse_loss(recon, img)
            loss = l_contrast + cfg.recon_weight * l_recon
            opt.zero_grad(); loss.backward(); opt.step()
            tot_c += l_contrast.item(); tot_r += l_recon.item(); n += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == cfg.epochs:
            r1_i2t, r1_t2i, *_ = retrieval_metrics(model, val_dl, device)
            print(f"epoch {epoch:3d} | contrast {tot_c/n:.3f} | recon {tot_r/n:.4f} "
                  f"| R@1 img→txt {r1_i2t:.3f} | R@1 txt→img {r1_t2i:.3f}")

    # final report + artifacts
    r1_i2t, r1_t2i, zi, zt, lab = retrieval_metrics(model, val_dl, device)
    print("\n=== RESULT ===")
    print(f"class-level R@1 image→text: {r1_i2t:.3f}")
    print(f"class-level R@1 text→image: {r1_t2i:.3f}")
    print(f"chance: {1/cfg.num_classes:.3f}")
    verdict = "PASS — alignment is forming, core claim holds at small scale" \
        if min(r1_i2t, r1_t2i) > 0.5 else \
        "WEAK — near chance; fix the core before adding any machinery"
    print(f"verdict: {verdict}")

    class_names = [f"{c} {s}" for (c, s) in train_ds.classes]
    plot_latent_space(zi, zt, lab, class_names, os.path.join(OUT, "latent_space.png"))
    plot_reconstructions(model, val_dl, device, os.path.join(OUT, "recon.png"))

    # save checkpoint so Day 2 (fragment store) can reuse the learned encoders
    ckpt_path = os.path.join(OUT, "worldfield.pt")
    torch.save({"model": model.state_dict(), "vocab_size": vocab_size,
                "class_names": class_names}, ckpt_path)
    print(f"\nartifacts: {OUT}/latent_space.png  {OUT}/recon.png  {ckpt_path}")


if __name__ == "__main__":
    main()
