"""Sample with one or more samplers and score the results.

    python evaluate.py --config configs/cityscapes.yaml --preset table1
    python evaluate.py --config configs/cityscapes.yaml --runs dual:3 sde:1000 ode:1000

A run is written method:steps[:schedule], schedule in {strided, early_stop}.
Every run is sampled `--trials` times with different seeds on the same inputs.

Reported per run:
  FID, LPIPS, PSNR, SSIM      mean over the test set (trial 0 for FID)
  psnr_std / ssim_std         std over trials of the test-set mean (one reading
                              of the paper's "five sampling trials" std)
  pixel_std                   mean per-pixel std across trials in [0, 1] units
                              (the quantity behind Fig. 3 / Fig. 8)
  mode_psnr                   ambiguous toy only: PSNR to the closest valid output
Method "input:1" scores the source image itself (bicubic baseline for SR).
Results go to <out_dir>/eval/<preset or 'custom'>/results.json and results.md.
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from dualbridge import BrownianBridge, sample
from dualbridge.data import build_dataset
from dualbridge.metrics import LPIPS, fid_from_dirs, psnr, ssim
from dualbridge.utils import load_config, load_network, save_grid, save_images

PRESETS = {
    # Table 1: effect of the number of sampling steps (paper: 1000 / 200 / 10 / 3)
    "table1": ["dual:1000", "dual:200", "dual:10", "dual:3"],
    # Table 2 / Fig. 3: Dual-approx vs the same bridge with SDE and PF-ODE samplers
    "table2": ["sde:1000", "ode:1000", "dual:3"],
    # Gap checks (not in the paper):
    #  - is the second network needed? (single derives z from eps_theta)
    #  - is "3 steps" really early stopping, i.e. nearly a one-shot regression?
    "gaps": ["dual:3", "single:3", "dual:3:early_stop", "dual:1", "sde:3", "ode:3",
             "dual:200", "single:200"],
    "quick": ["dual:3", "sde:50", "ode:50"],
    # Super-resolution on the scikit-learn photos (cf. Table 4), bicubic input as baseline
    "sr": ["input:1", "dual:1", "dual:3:early_stop", "dual:3", "dual:200", "single:200",
           "sde:200", "ode:200", "sde:1000", "ode:1000"],
    # Ambiguous toy: 4 valid outputs per input; does each sampler pick one or average them?
    "ambiguous": ["dual:1", "dual:3:early_stop", "dual:3", "dual:200", "single:200",
                  "sde:1000", "ode:1000"],
}


def parse_run(s):
    parts = s.split(":")
    return parts[0], int(parts[1]), parts[2] if len(parts) > 2 else "strided"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--num_test", type=int, default=0, help="limit test images (0 = all)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--fid", action="store_true", help="compute FID (needs pytorch-fid)")
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--tag", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = [parse_run(r) for r in (PRESETS[args.preset] if args.preset else []) + args.runs]
    assert runs, "give --preset or --runs"

    fwd = load_network(cfg, os.path.join(cfg["out_dir"], "forward", args.ckpt), device)
    rev_path = os.path.join(cfg["out_dir"], "reverse", args.ckpt)
    rev = load_network(cfg, rev_path, device) if os.path.exists(rev_path) else None
    bridge = BrownianBridge(cfg["model"]["num_timesteps"])
    cond = cfg["model"].get("condition", True)
    clip = cfg["model"].get("clip_denoised", True)

    ds = build_dataset(cfg, "test")
    if args.num_test:
        ds = Subset(ds, range(min(args.num_test, len(ds))))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    base = ds.dataset if isinstance(ds, Subset) else ds
    lp = LPIPS(device)

    tag = args.tag or args.preset or "custom"
    root = os.path.join(cfg["out_dir"], "eval", tag)
    gt_dir = os.path.join(cfg["out_dir"], "eval", "gt")
    if args.fid and not os.path.isdir(gt_dir):
        i = 0
        for x0, _ in dl:
            save_images(x0, gt_dir, [f"{i + k:05d}.png" for k in range(len(x0))]); i += len(x0)

    results = []
    for method, steps, sched in runs:
        if method == "dual" and rev is None:
            print("skipping dual: no reverse checkpoint"); continue
        name = f"{method}_{steps}" + ("" if sched == "strided" else f"_{sched}")
        print(f"== {name}")
        per_trial = {"psnr": [], "ssim": [], "lpips": [], "mode_psnr": []}
        pix_std, n_imgs, t_start = 0.0, 0, time.time()
        first_batch = None
        idx = 0
        for bi, (x0, y) in enumerate(dl):
            x0, y = x0.to(device), y.to(device)
            modes = None
            if getattr(base, "ambiguous", False):
                modes = torch.stack([base.modes(k) for k in range(idx, idx + len(x0))]).to(device)
            outs = []
            for trial in range(args.trials):
                g = torch.Generator(device=device).manual_seed(1000 * trial + bi)
                outs.append(sample(bridge, y, fwd, rev, method, steps, sched, clip, cond, g))
            stack = torch.stack(outs)                              # (trials, B, C, H, W)
            pix_std += ((stack + 1) / 2).std(0, unbiased=False).mean((1, 2, 3)).sum().item()
            n_imgs += len(x0)
            for trial, o in enumerate(outs):
                if len(per_trial["psnr"]) <= trial:
                    for k in per_trial: per_trial[k].append([])
                per_trial["psnr"][trial].append(psnr(o, x0).cpu())
                per_trial["ssim"][trial].append(ssim(o, x0).cpu())
                l = lp(o, x0)
                if l is not None:
                    per_trial["lpips"][trial].append(l.cpu())
                if modes is not None:
                    # PSNR to the closest of the valid outputs (ambiguous toy task)
                    best = torch.stack([psnr(o, modes[:, m]) for m in range(modes.shape[1])]).max(0).values
                    per_trial["mode_psnr"][trial].append(best.cpu())
            save_images(outs[0], os.path.join(root, name), [f"{idx + k:05d}.png" for k in range(len(x0))])
            idx += len(x0)
            if first_batch is None:
                first_batch = (y.cpu(), [o.cpu() for o in outs[:3]], x0.cpu())
        elapsed = time.time() - t_start

        def agg(key):
            if not per_trial[key] or not per_trial[key][0]:
                return None, None
            means = torch.stack([torch.cat(v).mean() for v in per_trial[key]])
            return means.mean().item(), means.std(unbiased=False).item()

        r = {"run": name, "method": method, "steps": steps, "schedule": sched, "trials": args.trials,
             "n_images": n_imgs, "seconds": round(elapsed, 1)}
        for key in ("psnr", "ssim", "lpips", "mode_psnr"):
            r[key], r[key + "_std"] = agg(key)
        r["pixel_std"] = pix_std / n_imgs
        r["fid"] = fid_from_dirs(os.path.join(root, name), gt_dir, device) if args.fid else None
        results.append(r)
        print(json.dumps(r))
        yb, ob, xb = first_batch
        save_grid([yb] + ob + [xb], os.path.join(root, f"grid_{name}.png"))

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    fmt = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
    lines = ["| run | FID | LPIPS | PSNR | SSIM % | nearest-mode PSNR | pixel std | sec |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['run']} | {fmt(r['fid'], 2)} | {fmt(r['lpips'])} | "
                     f"{fmt(r['psnr'], 2)} ± {fmt(r['psnr_std'], 2)} | "
                     f"{fmt(r['ssim'] * 100, 1)} ± {fmt(r['ssim_std'] * 100, 2)} | "
                     f"{fmt(r['mode_psnr'], 2)} | {fmt(r['pixel_std'], 4)} | {r['seconds']} |")
    with open(os.path.join(root, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
