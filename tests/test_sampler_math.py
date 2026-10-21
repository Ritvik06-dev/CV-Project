"""Checks of the sampler maths with oracle networks (no training needed).

    python -m pytest tests -q        or        python tests/test_sampler_math.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dualbridge import BrownianBridge, make_schedule, sample  # noqa: E402


def oracle(x0_true):
    """eps_theta that knows the true x0 and Z_phi that knows the true z."""
    bb = BrownianBridge(1000)
    fwd = lambda x, t, c: x - x0_true
    rev = lambda x, t, c: bb.z_from_x0(x, c, x0_true, t)
    return bb, fwd, rev


def test_schedules():
    assert make_schedule(1000, 3) == [1000, 667, 333]
    assert make_schedule(1000, 3, "early_stop") == [1000, 999, 998]
    assert make_schedule(1000, 1000) == list(range(1000, 0, -1))
    assert make_schedule(10, 50) == list(range(10, 0, -1))


def test_oracle_recovers_x0():
    torch.manual_seed(0)
    x0, y = torch.rand(2, 3, 8, 8) * 2 - 1, torch.rand(2, 3, 8, 8) * 2 - 1
    bb, fwd, rev = oracle(x0)
    for method in ("dual", "single", "sde", "ode"):
        out = sample(bb, y, fwd, rev, method, steps=20, clip=False)
        assert torch.allclose(out, x0, atol=1e-4), method


def test_unit_step_matches_algorithm3_line8():
    """Our generalised update with dt = 1 equals line 8 of Algorithm 3."""
    torch.manual_seed(0)
    bb = BrownianBridge(1000)
    x_t, y, x0h, z = (torch.randn(1, 3, 4, 4) for _ in range(4))
    t = 500
    tt, ss = torch.tensor([t]), torch.tensor([t - 1])
    u_hi = bb.q_sample(x0h, y, tt, z) - y
    u_lo = bb.q_sample(x0h, y, ss, z) - y
    line8 = (1 - 1 / t) * x_t + y / (t - 1) - x0h / (t * (t - 1)) - u_hi + t / (t - 1) * u_lo
    dt, s = 1, t - 1
    dW = u_hi - (1 + dt / s) * u_lo + (dt / s) * (x0h - y)
    ours = (1 - dt / t) * x_t + (dt / t) * x0h - dW
    assert torch.allclose(line8, ours, atol=1e-5)


def test_dual_update_keeps_bridge_mean():
    """With perfect x0 and z, the update lands on the bridge mean at s exactly;
    only the noise coefficient drifts (a property of Eq. 10, not a bug here)."""
    bb = BrownianBridge(1000)
    x0, y, z = torch.ones(1, 1, 1, 1), -torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)
    for t, s in [(1000, 667), (667, 333), (500, 499)]:
        tt, ss = torch.tensor([t]), torch.tensor([s])
        x_t = bb.q_sample(x0, y, tt, z)
        u_hi, u_lo = bb.q_sample(x0, y, tt, z) - y, bb.q_sample(x0, y, ss, z) - y
        dt = t - s
        x_s = (1 - dt / t) * x_t + (dt / t) * x0 - (u_hi - (1 + dt / s) * u_lo + (dt / s) * (x0 - y))
        assert torch.allclose(x_s, bb.q_sample(x0, y, ss, z), atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
