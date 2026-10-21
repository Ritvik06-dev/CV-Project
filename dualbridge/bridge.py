"""Brownian bridge between the target image x0 (t = 0) and the source image y (t = T).

Paper notation (Sec. 3.2 / 3.4, with g(t) = 1):
    m_t = t / T
    B(t) = sqrt(m_t (1 - m_t))                      (std of the bridge noise)
    x_t = (1 - m_t) x0 + m_t y + B(t) z,  z ~ N(0, I)          (Eq. 3)

Two approximators are trained on exactly this marginal:
    forward  eps_theta(x_t, t) ~= x_t - x0          (Eq. 7, objective "grad")
    reverse  Z_phi(x_t, t)     ~= z                 (Eq. 8, objective "noise")
"""
import torch
import torch.nn.functional as F


class BrownianBridge:
    def __init__(self, num_timesteps=1000):
        self.T = num_timesteps

    def m(self, t):
        return t.float() / self.T

    def B(self, t):
        m = self.m(t)
        return torch.sqrt((m * (1.0 - m)).clamp(min=0.0))

    @staticmethod
    def _b(v, ref):
        """Broadcast a per-sample vector (or scalar) to the image shape."""
        if not torch.is_tensor(v):
            v = torch.tensor(v, device=ref.device, dtype=ref.dtype)
        v = v.to(ref.device, ref.dtype)
        if v.dim() == 0:
            v = v.expand(ref.shape[0])
        return v.view(-1, *([1] * (ref.dim() - 1)))

    def q_sample(self, x0, y, t, z):
        m, b = self._b(self.m(t), x0), self._b(self.B(t), x0)
        return (1.0 - m) * x0 + m * y + b * z

    # ---- converting network outputs -------------------------------------------------
    def x0_from_grad(self, x_t, grad):
        """Eq. 13: x0_hat = x_t - eps_theta(x_t, t)."""
        return x_t - grad

    def z_from_x0(self, x_t, y, x0_hat, t):
        """Noise implied by an x0 estimate (used by the single-network ablation)."""
        m, b = self._b(self.m(t), x_t), self._b(self.B(t), x_t)
        return (x_t - (1.0 - m) * x0_hat - m * y) / b.clamp(min=1e-8)

    # ---- training losses -------------------------------------------------------------
    def training_loss(self, net, x0, y, role, loss_type="l1", cond=True):
        """One training step for either approximator.

        role="forward": Algorithm 1, target x_t - x0.
        role="reverse": Algorithm 2, target z.
        The official code samples t uniformly from {0, ..., T-1} for both.
        """
        n = x0.shape[0]
        t = torch.randint(0, self.T, (n,), device=x0.device)
        z = torch.randn_like(x0)
        x_t = self.q_sample(x0, y, t, z)
        target = (x_t - x0) if role == "forward" else z
        pred = net(x_t, t, y if cond else None)
        if loss_type == "l1":
            return (pred - target).abs().mean()
        return F.mse_loss(pred, target)
