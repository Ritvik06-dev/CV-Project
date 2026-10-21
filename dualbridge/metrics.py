"""Evaluation metrics. Images are float tensors in [-1, 1], shape (N, 3, H, W).

PSNR and SSIM are implemented here (no extra dependencies).
LPIPS uses the `lpips` package and FID uses `pytorch-fid`; both are optional and
return None if the package is missing.
"""
import torch
import torch.nn.functional as F


def _01(x):
    return ((x.clamp(-1, 1) + 1) / 2).float()


def psnr(pred, gt):
    """Per-image PSNR in dB on [0, 1] RGB."""
    mse = ((_01(pred) - _01(gt)) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp(min=1e-10))


def _gauss(size=11, sigma=1.5, ch=3, device="cpu"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    return g.expand(ch, 1, size, size).contiguous()


def ssim(pred, gt):
    """Per-image SSIM (Wang et al. 2004, 11x11 Gaussian window) on [0, 1] RGB."""
    x, y = _01(pred), _01(gt)
    w = _gauss(ch=x.shape[1], device=x.device)
    conv = lambda z: F.conv2d(z, w, groups=x.shape[1])
    mx, my = conv(x), conv(y)
    sxx, syy, sxy = conv(x * x) - mx ** 2, conv(y * y) - my ** 2, conv(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return m.flatten(1).mean(1)


class LPIPS:
    def __init__(self, device):
        try:
            import lpips
            self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception:
            self.net = None

    @torch.no_grad()
    def __call__(self, pred, gt):
        if self.net is None:
            return None
        return self.net(pred.clamp(-1, 1), gt.clamp(-1, 1)).flatten()


def fid_from_dirs(dir_a, dir_b, device="cuda", batch_size=32):
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except Exception:
        return None
    return float(calculate_fid_given_paths([dir_a, dir_b], batch_size, device, 2048, num_workers=0))
