"""Slide figures for a trained config. Writes to <out_dir>/figures/.

    python make_figures.py --config configs/toy.yaml
    python make_figures.py --config configs/toy.yaml --only variance trajectory
    python make_figures.py --config configs/toy.yaml --only steps --replot   (no resampling)

forward_bridge.png   the Brownian bridge from target x0 (t=0) to source y (t=T)
variance.png         one input sampled 5 times by each sampler + per-pixel std map
trajectory.png       x_t and x0_hat along a 1000-step Dual-approx and SDE run
steps_quality.png    PSNR (or nearest-mode PSNR) vs number of sampling steps
steps_variance.png   output std across trials vs number of sampling steps
steps_sweep.csv      the numbers behind both step plots
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from dualbridge import BrownianBridge, sample  # noqa: E402
from dualbridge.data import build_dataset  # noqa: E402
from dualbridge.metrics import psnr  # noqa: E402
from dualbridge.utils import load_config, load_network  # noqa: E402

# reference categorical palette, fixed order (validated light-mode slots 1-5)
COLORS = {"dual": "#2a78d6", "dual_early": "#eb6834", "single": "#1baf7a", "sde": "#eda100", "ode": "#e87ba4"}
LABELS = {"dual": "Dual-approx (strided)", "dual_early": "Dual-approx (early stop)",
          "single": "Single network", "sde": "SDE sampler", "ode": "ODE sampler"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def img(x):
    return ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def image_grid(rows, row_labels, col_labels, path, std_maps=None):
    n_rows, n_cols = len(rows), len(rows[0]) + (1 if std_maps is not None else 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.45 * n_cols + 1.6, 1.5 * n_rows + 0.4), squeeze=False)
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c < len(row):
                if row[c] is not None:
                    ax.imshow(img(row[c]), interpolation="nearest")
            else:
                ax.imshow(std_maps[r], cmap="Greys", vmin=0, vmax=max(0.05, float(std_maps.max())),
                          interpolation="nearest")
            if r == 0:
                ax.set_title(col_labels[c], fontsize=9, color=INK)
        axes[r][0].set_ylabel(row_labels[r], fontsize=9, color=INK, rotation=0, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", type=int, default=3, help="test image used for single-image figures")
    ap.add_argument("--num_test", type=int, default=32, help="images used for the step sweep")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=["bridge", "variance", "trajectory", "steps"])
    ap.add_argument("--replot", action="store_true", help="redraw step plots from steps_sweep.csv")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(cfg["out_dir"], "figures")
    os.makedirs(out, exist_ok=True)
    fwd = load_network(cfg, os.path.join(cfg["out_dir"], "forward", "final.pt"), device)
    rev = load_network(cfg, os.path.join(cfg["out_dir"], "reverse", "final.pt"), device)
    bridge = BrownianBridge(cfg["model"]["num_timesteps"])
    T = bridge.T
    ds = build_dataset(cfg, "test")
    x0, y = (v.unsqueeze(0).to(device) for v in ds[args.index])

    def run(method, steps, sched="strided", yy=y, seed=0, trace=False):
        g = torch.Generator(device=device).manual_seed(seed)
        return sample(bridge, yy, fwd, rev, method, steps, sched, True, True, g, return_trace=trace)

    if "bridge" in args.only:
        ts = [0, 200, 400, 600, 800, 1000]
        g = torch.Generator(device=device).manual_seed(0)
        z = torch.randn(x0.shape, device=device, generator=g)
        frames = [bridge.q_sample(x0, y, torch.tensor([t], device=device), z)[0] for t in ts]
        image_grid([frames], [""], [f"t = {t}" for t in ts], os.path.join(out, "forward_bridge.png"))

    if "variance" in args.only:
        runs = [("sde", 1000, "strided", "SDE, 1000 steps"), ("ode", 1000, "strided", "ODE, 1000 steps"),
                ("dual", 3, "strided", "Dual, 3 strided"), ("dual", 3, "early_stop", "Dual, 3 early-stop"),
                ("dual", 200, "strided", "Dual, 200 steps")]
        rows, stds = [], []
        for m, k, sch, _ in runs:
            outs = [run(m, k, sch, seed=s)[0] for s in range(5)]
            rows.append([y[0]] + outs + [x0[0]])
            stds.append(((torch.stack(outs) + 1) / 2).std(0, unbiased=False).mean(0).cpu())
        image_grid(rows, [r[3] for r in runs], ["input y"] + [f"trial {i + 1}" for i in range(5)] + ["ground truth", "pixel std"],
                   os.path.join(out, "variance.png"), torch.stack(stds))

    if "trajectory" in args.only:
        show = [1000, 800, 600, 400, 200, 50, 1]
        rows = []
        for m in ("dual", "sde"):
            _, tr = run(m, 1000, trace=True)
            by_t = {t: (xt, xh) for t, xt, xh in tr}
            rows.append([by_t[t][0][0] for t in show])
            rows.append([by_t[t][1][0] for t in show])
        image_grid(rows, ["Dual: x_t", "Dual: x0_hat", "SDE: x_t", "SDE: x0_hat"], [f"t = {t}" for t in show],
                   os.path.join(out, "trajectory.png"))

    if "steps" in args.only:
        steps_list = [1, 2, 3, 5, 10, 20, 50, 100, 200, 1000]
        series = {"dual": ("dual", "strided"), "dual_early": ("dual", "early_stop"), "single": ("single", "strided"),
                  "sde": ("sde", "strided"), "ode": ("ode", "strided")}
        n = min(args.num_test, len(ds))
        xs, ys = zip(*[ds[i] for i in range(n)])
        X0, Y = torch.stack(xs).to(device), torch.stack(ys).to(device)
        ambiguous = getattr(ds, "ambiguous", False)
        modes = torch.stack([ds.modes(i) for i in range(n)]).to(device) if ambiguous else None
        csv_path = os.path.join(out, "steps_sweep.csv")
        rows = []
        if args.replot:
            with open(csv_path) as f:
                rows = [{"series": r["series"], "steps": int(r["steps"]), "psnr": float(r["psnr"]),
                         "mode_psnr": float(r["mode_psnr"]) if r["mode_psnr"] else None,
                         "pixel_std": float(r["pixel_std"])} for r in csv.DictReader(f)]
        for key, (m, sch) in ([] if args.replot else series.items()):
            for k in steps_list:
                if sch == "early_stop" and k >= T:
                    continue
                outs = [run(m, k, sch, yy=Y, seed=s) for s in range(args.trials)]
                q = torch.stack([psnr(o, X0).mean() for o in outs]).mean().item()
                mq = None
                if modes is not None:
                    mq = torch.stack([torch.stack([psnr(o, modes[:, j]) for j in range(modes.shape[1])]).max(0).values.mean()
                                      for o in outs]).mean().item()
                sd = ((torch.stack(outs) + 1) / 2).std(0, unbiased=False).mean().item()
                rows.append({"series": key, "steps": k, "psnr": q, "mode_psnr": mq, "pixel_std": sd})
                print(rows[-1], flush=True)
        if not args.replot:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)

        metric = "mode_psnr" if ambiguous else "psnr"
        ylabel = "PSNR to nearest valid output (dB)" if ambiguous else "PSNR to ground truth (dB)"
        for fname, key, yl in (("steps_quality.png", metric, ylabel),
                               ("steps_variance.png", "pixel_std", "Std across trials (pixel units, 0-1)")):
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            style(ax)
            for s in series:
                pts = [(r["steps"], r[key]) for r in rows if r["series"] == s]
                ax.plot(*zip(*pts), color=COLORS[s], linewidth=2, marker="o", markersize=5,
                        markeredgecolor="white", markeredgewidth=1.2, label=LABELS[s])
            ax.set_xscale("log")
            ax.set_xlabel("Sampling steps (network evaluations)", fontsize=10, color=INK)
            ax.set_ylabel(yl, fontsize=10, color=INK)
            ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), dpi=200)
            plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main()
