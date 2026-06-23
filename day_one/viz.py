"""Visualization of the shared latent space (plan §24: without visualization you
won't know if the system is learning anything). 2D projection via PCA (no extra
deps); falls back gracefully if matplotlib is missing."""
import torch


def _project_2d(x):
    """PCA to 2D using torch SVD — no sklearn/umap dependency."""
    x = x - x.mean(dim=0, keepdim=True)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    return x @ v[:2].t()


def plot_latent_space(zi, zt, lab, class_names, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping latent_space.png)")
        return

    # project image + text embeddings together so the two modalities share axes
    both = torch.cat([zi, zt], dim=0)
    p = _project_2d(both)
    pi, pt = p[: zi.size(0)], p[zi.size(0):]

    n_cls = len(class_names)
    cmap = plt.get_cmap("tab20", n_cls)
    fig, ax = plt.subplots(figsize=(9, 8))
    for c in range(n_cls):
        mi = lab == c
        ax.scatter(pi[mi, 0], pi[mi, 1], color=cmap(c), marker="o", s=14, alpha=0.6)
        ax.scatter(pt[mi, 0], pt[mi, 1], color=cmap(c), marker="x", s=30, alpha=0.9)
    # legend: color = class, marker = modality
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=cmap(c), marker="o", linestyle="", label=class_names[c])
               for c in range(n_cls)]
    handles += [Line2D([], [], color="k", marker="o", linestyle="", label="image"),
                Line2D([], [], color="k", marker="x", linestyle="", label="text")]
    ax.legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Shared latent space (PCA-2D)  •  o = image, x = text, color = class")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reconstructions(model, loader, device, path, n=8):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping recon.png)")
        return

    model.eval()
    img, ids, _ = next(iter(loader))
    img = img.to(device)[:n]
    with torch.no_grad():
        zi = model.encode_image(img)
        recon = model.img_dec(zi)
    img = img.cpu().permute(0, 2, 3, 1).numpy()
    recon = recon.cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 3))
    for j in range(n):
        axes[0, j].imshow(img[j]); axes[0, j].axis("off")
        axes[1, j].imshow(recon[j]); axes[1, j].axis("off")
    axes[0, 0].set_ylabel("input"); axes[1, 0].set_ylabel("recon")
    fig.suptitle("Image reconstruction from shared latent")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
