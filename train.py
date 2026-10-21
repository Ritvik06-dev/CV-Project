"""Train one of the two approximators.

    python train.py --config configs/cityscapes.yaml --role forward
    python train.py --config configs/cityscapes.yaml --role reverse

forward = eps_theta (Algorithm 1), reverse = Z_phi (Algorithm 2). The two are
trained independently, as in the paper; run them one after the other (or on two
GPUs). Checkpoints go to <out_dir>/<role>/.
"""
import argparse
import csv
import os
import time

import torch
from torch.utils.data import DataLoader

from dualbridge import BrownianBridge
from dualbridge.data import build_dataset
from dualbridge.utils import EMA, build_unet, load_config, save_grid, seed_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--role", choices=["forward", "reverse"], required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.iters=1000")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    tc = cfg["train"]
    seed_all(tc.get("seed", 0) + (0 if args.role == "forward" else 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(cfg["out_dir"], args.role)
    os.makedirs(out, exist_ok=True)

    ds = build_dataset(cfg, "train")
    dl = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                    num_workers=tc.get("num_workers", 2), pin_memory=device == "cuda",
                    persistent_workers=tc.get("num_workers", 2) > 0)
    bridge = BrownianBridge(cfg["model"]["num_timesteps"])
    cond = cfg["model"].get("condition", True)

    net = build_unet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=tc["lr"])
    ema = EMA(net, tc.get("ema_decay", 0.999))
    step = 0
    ckpt_path = os.path.join(out, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(st["model"]); ema.shadow.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); step = st["step"]
        print(f"resumed from step {step}")

    print(f"[{args.role}] {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params, "
          f"{len(ds)} training pairs, device={device}")

    amp = device == "cuda" and tc.get("amp", True)
    accum = tc.get("grad_accum", 1)
    log = open(os.path.join(out, "log.csv"), "a", newline="")
    writer = csv.writer(log)
    preview = next(iter(DataLoader(build_dataset(cfg, "test"), batch_size=8)))

    def save(name):
        torch.save({"model": net.state_dict(), "ema": ema.shadow.state_dict(), "opt": opt.state_dict(),
                    "step": step, "config": cfg, "role": args.role}, os.path.join(out, name))

    net.train()
    t0, running = time.time(), 0.0
    data_iter = iter(dl)
    while step < tc["iters"]:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            try:
                x0, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                x0, y = next(data_iter)
            x0, y = x0.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = bridge.training_loss(net, x0, y, args.role, cfg["model"].get("loss", "l1"), cond)
            (loss / accum).backward()
            running += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(net.parameters(), tc.get("grad_clip", 1.0))
        opt.step()
        ema.update(net)
        step += 1

        if step % tc.get("log_every", 100) == 0:
            avg = running / tc.get("log_every", 100)
            running = 0.0
            it_s = tc.get("log_every", 100) / (time.time() - t0)
            t0 = time.time()
            writer.writerow([step, f"{avg:.5f}"]); log.flush()
            eta_min = (tc["iters"] - step) / it_s / 60
            print(f"step {step:>7}  loss {avg:.4f}  {it_s:.2f} it/s  eta {eta_min:.0f} min", flush=True)

        if step % tc.get("preview_every", 2000) == 0 and args.role == "forward":
            # one-step x0 prediction from x_T = y, a quick visual check of eps_theta
            with torch.no_grad():
                yp = preview[1].to(device)
                t = torch.full((yp.shape[0],), bridge.T, device=device, dtype=torch.long)
                x0_hat = (yp - ema.shadow(yp, t, yp if cond else None)).clamp(-1, 1)
            save_grid([yp.cpu(), x0_hat.cpu(), preview[0]], os.path.join(out, f"preview_{step:07d}.png"))

        if step % tc.get("save_every", 5000) == 0:
            save("last.pt")
    save("last.pt")
    save("final.pt")
    log.close()


if __name__ == "__main__":
    main()
