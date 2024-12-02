"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
import torch
import torch.nn as nn

import manifm.model.diffeq_layers as diffeq_layers
from manifm.model.actfn import Sine, Softplus
from manifm.manifolds.mesh import Mesh, closest_point, face_normal
from manifm.manifolds import SPD


ACTFNS = {
    "swish": diffeq_layers.TimeDependentSwish,
    "sine": Sine,
    "srelu": Softplus,
}

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def MLP(d_in, d_out=None, d_model=256, num_layers=6, actfn="swish"):
    assert num_layers > 1, "No weak linear nets here"
    d_out = d_in if d_out is None else d_out
    actfn = Swish()
    layers = [nn.Linear(d_in, d_model)]

    for _ in range(num_layers - 2):
        layers.append(actfn)
        layers.append(nn.Linear(d_model, d_model))
    layers.append(actfn)
    layers.append(nn.Linear(d_model, d_out))
    return nn.Sequential(*layers)

def cond_MLP(d_in, d_out=None, d_model=256, d_cond=512, num_layers=6, actfn="swish"):
    assert num_layers > 1, "No weak linear nets here"
    d_out = d_in if d_out is None else d_out
    actfn = Swish()
    layers = [nn.Linear(d_in+d_cond, d_model)]

    for _ in range(num_layers - 2):
        layers.append(actfn)
        layers.append(nn.Linear(d_model, d_model))
    layers.append(actfn)
    layers.append(nn.Linear(d_model, d_out))
    return nn.Sequential(*layers)

def tMLP(d_in, d_out=None, d_model=256, num_layers=6, actfn="swish", fourier=None):
    assert num_layers > 1, "No weak linear nets here"
    d_out = d_in if d_out is None else d_out
    actfn = ACTFNS[actfn]
    if fourier:
        layers = [
            diffeq_layers.diffeq_wrapper(
                PositionalEncoding(n_fourier_features=fourier)
            ),
            diffeq_layers.ConcatLinear_v2(d_in * fourier * 2, d_model),
        ]
    else:
        layers = [diffeq_layers.ConcatLinear_v2(d_in, d_model)]

    for _ in range(num_layers - 2):
        layers.append(actfn(d_model))
        layers.append(diffeq_layers.ConcatLinear_v2(d_model, d_model))
    layers.append(actfn(d_model))
    layers.append(diffeq_layers.ConcatLinear_v2(d_model, d_out))
    return diffeq_layers.SequentialDiffEq(*layers)


class PositionalEncoding(nn.Module):
    """Assumes input is in [0, 2pi]."""

    def __init__(self, n_fourier_features):
        super().__init__()
        self.n_fourier_features = n_fourier_features

    def forward(self, x):
        feature_vector = [
            torch.sin((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        feature_vector += [
            torch.cos((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        return torch.cat(feature_vector, dim=-1)


class LatentRectifiedFlow(nn.Module):
    def __init__(
        self,
        d_in,
        d_latent,
        d_model,
        num_ae_layers,
        num_fm_layers,
        actfn,
        fourier,
        manifold,
        has_cond=False,
    ):
        super().__init__()
        if has_cond:
            self.encoder = cond_MLP(
                d_in, d_latent, d_model, 512,
                num_ae_layers//2, actfn)
            self.decoder = MLP(
                d_latent, d_in, d_model,
                num_ae_layers//2, actfn)
            self.latent_vecfield = tMLP(
                d_latent, d_latent, d_model,
                num_fm_layers, actfn, fourier)
            self.manifold = manifold
        else:
            self.encoder = MLP(
                d_in, d_latent, d_model,
                num_ae_layers//2, actfn)
            self.decoder = MLP(
                d_latent, d_in, d_model,
                num_ae_layers//2, actfn)
            self.latent_vecfield = tMLP(
                d_latent, d_latent, d_model,
                num_fm_layers, actfn, fourier)
            self.manifold = manifold

    def forward(self, t, x_or_l, cond=None, projl=True, vecfield=True, recon=True):
        if cond is not None:
            x_or_l = torch.cat([x_or_l, cond], dim=-1)

        # Batchify
        has_batch = x_or_l.ndim > 1
        if not has_batch:
            x_or_l = x_or_l.reshape(1, -1)
            t = t.reshape(-1)

        if projl:
            x = x_or_l
            x = self._apply_manifold_constraint(x)
            l = self.encoder(x)
            # l_ = l.detach()
        else:
            l = x_or_l
            # l_ = l      # NOTE: Required for computation of divergence.

        # # NOTE: detach latent in flow matching loss.
        # v = self.latent_vecfield(t, l_) if vecfield else None
        v = self.latent_vecfield(t, l) if vecfield else None
        x_hat = self.decoder(l) if recon else None

        # Unbatchify
        if not has_batch:
            x_hat = x_hat[0] if recon else None
            v = v[0] if vecfield else None

        return l, v, x_hat

    # def encode(self, t, x):
    #     x = self._apply_manifold_constraint(x)
    #     return self.encoder(t, x)

    # def decode(self, t, l):
    #     return self.decoder(t, l)

    # def compute_latent_vecfield(self, t, l):
    #     return self.latent_vecfield(t, l)

    def _apply_manifold_constraint(self, x):
        if isinstance(self.manifold, Mesh):
            # Memory-efficient implementation for meshes.
            with torch.no_grad():
                _, f_idx = closest_point(x, self.manifold.v, self.manifold.f)
                vs = self.manifold.v[self.manifold.f[f_idx]]
                n = face_normal(a=vs[:, 0], b=vs[:, 1], c=vs[:, 2])
            x = x + (n * (vs[:, 0] - x)).sum(-1, keepdim=True) * n
        if isinstance(self.manifold, SPD):
            # projx is expensive and we can just skip it since it doesn't affect divergence.
            pass
        else:
            x = self.manifold.projx(x)
        return x


@torch.no_grad()
def latent_odeint(model, x, t):
    if not isinstance(model, LatentRectifiedFlow):
        raise ValueError("model must be LatentRectifiedFlow.")
    num_steps = len(t)-1
    xs = [x]
    for i in range(num_steps):
        ti, dt = t[i], t[i+1] - t[i]
        # l = model.encoder(ti, x)
        l = model.encoder(x)
        v = model.latent_vecfield(ti, l)

        # Extrapolate latent.
        l = l + v * dt
        # x = model.decoder(ti, l)
        x = model.decoder(l)
        xs.append(x)

    xs = torch.stack(xs, dim=0)
    return xs

@torch.no_grad()
def projx_latent_odeint(manifold, model, x, t, projx=True, local_coords=False):
    if not isinstance(model, LatentRectifiedFlow):
        raise ValueError("model must be LatentRectifiedFlow.")
    num_steps = len(t)-1
    xs = [x]
    for i in range(num_steps):
        ti, dt = t[i], t[i+1] - t[i]
        l = model.encoder(x)
        v = model.latent_vecfield(ti, l)

        # Extrapolate latent.
        l = l + v * dt
        x = model.decoder(l)
        if projx:
            x = manifold.projx(x)
        xs.append(x)

    xs = torch.stack(xs, dim=0)
    return xs


if __name__ == "__main__":
    print(diffeq_layers.ConcatLinear_v2(3, 64))

    import torch

    model = tMLP(d_in=3, d_model=64, num_layers=3)
    t = torch.randn(2, 1)
    x = torch.randn(2, 3)

    print(model(t, x))
