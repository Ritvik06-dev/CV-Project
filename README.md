# Dual-approx Bridge: re-implementation

Re-implementation of *Deterministic Image-to-Image Translation via Denoising Brownian
Bridge Models with Dual Approximators* (Xiao, Wang, He, Dong, CVPR 2025).

The authors' public repo (github.com/bohan95/dual-app-bridge) ships the two training
objectives and a standard SDE sampler, but not the dual-approximator sampler of
Algorithm 3. The sampler here is written from Eq. 9, 10, 12 and Algorithm 3 of the paper.

## Layout

```
dualbridge/bridge.py     Brownian bridge marginal and the two training losses
dualbridge/unet.py       time-conditioned UNet, y concatenated to x_t
dualbridge/samplers.py   dual (paper), single (ablation), sde, ode samplers + step schedules
dualbridge/data.py       paired datasets: pix2pix A|B, A/B folders, super-resolution, synthetic
dualbridge/metrics.py    PSNR, SSIM (built in), LPIPS and FID (optional packages)
dualbridge/utils.py      config loading, EMA, checkpoints, image saving
train.py                 trains the forward (eps_theta) or reverse (Z_phi) approximator
evaluate.py              samples with any set of samplers, N trials each, writes tables
make_figures.py          slide figures: bridge path, 5-trial variance grid, trajectories, step sweeps
configs/*.yaml           toy, toy_ambiguous, sr_sample (no download needed);
                         cityscapes, edges2handbags, sr_df2k (paper datasets)
scripts/download_data.py fetches the pix2pix cityscapes / edges2handbags archives
tests/                   sampler maths checked against oracle networks
```

## Running

```bash
pip install -r requirements.txt
python tests/test_sampler_math.py

# 1. pipeline check on synthetic data (minutes)
python train.py --config configs/toy.yaml --role forward
python train.py --config configs/toy.yaml --role reverse
python evaluate.py --config configs/toy.yaml --preset gaps --trials 5

# 2. toy task with 4 valid outputs per input (~15 min training)
python train.py --config configs/toy_ambiguous.yaml --role forward
python train.py --config configs/toy_ambiguous.yaml --role reverse
python evaluate.py --config configs/toy_ambiguous.yaml --preset ambiguous --trials 5 --batch_size 64

# 3. 4x super-resolution on the scikit-learn sample photos (~45 min training)
python train.py --config configs/sr_sample.yaml --role forward
python train.py --config configs/sr_sample.yaml --role reverse
python evaluate.py --config configs/sr_sample.yaml --preset sr --trials 3 --batch_size 48

# figures for any trained config -> runs/<name>/figures/
python make_figures.py --config configs/toy_ambiguous.yaml --index 5

# 4. Cityscapes label -> photo (needs the ~100 MB download, hours of training)
python scripts/download_data.py cityscapes
python train.py --config configs/cityscapes.yaml --role forward
python train.py --config configs/cityscapes.yaml --role reverse
python evaluate.py --config configs/cityscapes.yaml --preset table1 --fid
python evaluate.py --config configs/cityscapes.yaml --preset table2 --fid
python evaluate.py --config configs/cityscapes.yaml --preset gaps
```

Any config value can be overridden on the command line, e.g.
`python train.py --config configs/cityscapes.yaml --role forward data.image_size=128 train.iters=50000`.

Evaluation runs are written `method:steps[:schedule]`, e.g. `dual:3`, `sde:1000`,
`dual:3:early_stop`. Results land in `runs/<name>/eval/<preset>/results.md`.

## Choices that the paper leaves open

- `B(t)` is the bridge std `sqrt(m_t(1-m_t))`, as in the official code (the paper writes it as `G(t/T)`).
- Algorithm 3 divides by `t-1` and breaks at `t=1`; we stop at the last `t >= 1` and output `x0_hat` (Eq. 14-16).
- For strided schedules the step `t -> s` uses Eq. 9/10 with `dt = t - s`. With `dt = 1` it is line 8 of Algorithm 3 exactly (checked in tests).
- "3 sampling steps" is ambiguous: evenly strided (`dual:3`) or the first three unit steps from `T` (`dual:3:early_stop`). Both are available.
- The PF-ODE baseline is the deterministic bridge sampler (DDIM, eta = 0).
- The SR scale factor isn't given in the paper; the config uses 4x.
