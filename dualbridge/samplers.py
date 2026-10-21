"""Samplers for the Brownian bridge.

All samplers start at x_T = y and evaluate the networks at a decreasing list of
integer timesteps ts = [t_0 = T, t_1, ..., t_{K-1}] (K network evaluations).
After the last evaluation they return x0_hat = x_t - eps_theta(x_t, t) (Eq. 16).

dual    Dual-approx Bridge (Sec. 3.3, Eq. 9/10/12, Algorithm 3).
single  Same update, but z_hat is derived from eps_theta instead of a second
        network. Ablation for the paper's claim that one approximator "cannot
        serve this purpose".
sde     Euler-Maruyama discretisation of the reverse SDE, as in the official
        ABridge code: x_s = x_t - (dt/t) eps - sqrt(dt/T) * noise.
ode     Deterministic bridge sampler (DDIM with eta = 0), the standard
        PF-ODE-style sampler for Brownian bridges.

Generalised Dual-approx update for a step t -> s (dt = t - s, integers):
    x0_hat = x_t - eps_theta(x_t, t)
    z_hat  = N(0, I) if t == T else Z_phi(x_t, t)
    U_hi   = q(x0_hat, y, t, z_hat) - y
    U_lo   = q(x0_hat, y, s, z_hat) - y
    dW     = U_hi - (1 + dt/s) U_lo + (dt/s)(x0_hat - y)              (Eq. 10)
    x_s    = (1 - dt/t) x_t + (dt/t) x0_hat - dW                       (Eq. 9)
With dt = 1 this is exactly line 8 of Algorithm 3. Algorithm 3 as printed
loops down to t = 1, where line 8 divides by t - 1 = 0; we stop at the last
t >= 1 and output x0_hat instead, which is what Eq. 14-16 describe.
"""
import torch


def make_schedule(T, steps, mode="strided"):
    """Timesteps at which the networks are evaluated.

    strided     K evenly spaced points from T down to T/K (the usual skip-sampling).
    early_stop  The first K unit steps T, T-1, ..., T-K+1, then read off x0_hat.
                This is how Sec. 4.1.1 describes the step ablation ("sampling can
                be stopped at any time").
    """
    if mode == "early_stop":
        return list(range(T, T - steps, -1))
    if steps >= T:
        return list(range(T, 0, -1))
    ts = torch.linspace(T, 0, steps + 1)[:-1].round().long().tolist()
    out = []
    for t in ts:
        if t >= 1 and (not out or t < out[-1]):
            out.append(t)
    return out


def _full(t, x):
    return torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)


@torch.no_grad()
def sample(bridge, y, fwd, rev=None, method="dual", steps=3, schedule="strided",
           clip=True, cond=True, generator=None, return_trace=False):
    if method == "input":  # baseline: return the source image unchanged
        return (y.clone(), []) if return_trace else y.clone()
    T = bridge.T
    ts = make_schedule(T, steps, schedule)
    c = y if cond else None
    x = y.clone()
    trace = []

    def randn():
        return torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)

    for i, t in enumerate(ts):
        tt = _full(t, x)
        grad = fwd(x, tt, c)
        x0_hat = bridge.x0_from_grad(x, grad)
        if clip:
            x0_hat = x0_hat.clamp(-1, 1)
        if return_trace:
            trace.append((t, x.detach().cpu(), x0_hat.detach().cpu()))
        if i == len(ts) - 1:
            break
        s = ts[i + 1]
        dt = t - s

        if method in ("dual", "single"):
            if t == T:
                z_hat = randn()  # B(T) = 0, only enters through q(., s, z_hat)
            elif method == "dual":
                z_hat = rev(x, tt, c)
            else:
                z_hat = bridge.z_from_x0(x, y, x0_hat, tt)
            ss = _full(s, x)
            u_hi = bridge.q_sample(x0_hat, y, tt, z_hat) - y
            u_lo = bridge.q_sample(x0_hat, y, ss, z_hat) - y
            dW = u_hi - (1 + dt / s) * u_lo + (dt / s) * (x0_hat - y)
            x = (1 - dt / t) * x + (dt / t) * x0_hat - dW

        elif method == "sde":
            x = x - (dt / t) * (x - x0_hat) - (dt / T) ** 0.5 * randn()

        elif method == "ode":
            ss = _full(s, x)
            ms, bs = bridge.m(ss)[0].item(), bridge.B(ss)[0].item()
            mt, bt = bridge.m(tt)[0].item(), bridge.B(tt)[0].item()
            ratio = bs / bt if bt > 0 else 0.0
            x = (1 - ms) * x0_hat + ms * y + ratio * (x - (1 - mt) * x0_hat - mt * y)
        else:
            raise ValueError(f"unknown sampler {method}")

    return (x0_hat, trace) if return_trace else x0_hat
