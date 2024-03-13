"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import igl
import numpy as np
import torch
from sklearn.utils import shuffle as util_shuffle
import numpy as np

from manifm.manifolds.mesh import Mesh, sample_simplex_uniform
from manifm.dist import GaussianMM


def generate_rings(nsamples, seed=0):
    rng = np.random.RandomState(seed)

    n_samples4 = n_samples3 = n_samples2 = nsamples // 4
    n_samples1 = nsamples - n_samples4 - n_samples3 - n_samples2

    # so as not to have the first point = last point, we set endpoint=False
    linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
    linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
    linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
    linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

    circ4_x = np.cos(linspace4)
    circ4_y = np.sin(linspace4)
    circ3_x = np.cos(linspace4) * 0.75
    circ3_y = np.sin(linspace3) * 0.75
    circ2_x = np.cos(linspace2) * 0.5
    circ2_y = np.sin(linspace2) * 0.5
    circ1_x = np.cos(linspace1) * 0.25
    circ1_y = np.sin(linspace1) * 0.25

    X = (
        np.vstack(
            [
                np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
                np.hstack([circ4_y, circ3_y, circ2_y, circ1_y]),
            ]
        ).T
        * 3.0
    )
    X = util_shuffle(X, random_state=rng)

    # Add noise
    X = X + rng.normal(scale=0.05, size=X.shape)

    return X.astype("float32")


def create_simple_bunny(replace: bool = False):
    if not replace and os.path.exists("mesh/bunny_simple.npy"):
        print("mesh/bunny_simple.npy exists. Skipping.")
        return

    np.random.seed(777)

    v, f = igl.read_triangle_mesh("mesh/bunny_simp.obj")
    vi = v[234]
    x = np.random.randn(200000, 3) * 10.0 + vi
    x = igl.signed_distance(x, v, f)[2]
    with open("mesh/bunny_simple.npy", "wb") as f:
        np.save(f, x.astype(np.float32))


def create_eigfn(
    obj: str, idx: int, nsamples: int = 500000, upsample: int = 0, replace: bool = False
):
    if not replace and os.path.exists(f"mesh/{obj}_eigfn{idx:03d}.npy"):
        print(f"mesh/{obj}_eigfn{idx:03d}.npy exists. Skipping.")
        return

    np.random.seed(777)

    v, f = igl.read_triangle_mesh(f"../data/mesh/{obj}_simp.obj")

    v, f = torch.tensor(v), torch.tensor(f)
    mesh = Mesh(v, f, numeigs=idx + 20, upsample=upsample)

    vals = mesh.eigfns[:, idx].clamp(min=0.0000)
    vals = torch.mean(vals[mesh.f], dim=1)
    vals = vals * mesh.areas

    f_idx = torch.multinomial(vals, nsamples, replacement=True)
    barycoords = sample_simplex_uniform(
        2, (nsamples,), dtype=mesh.v.dtype, device=mesh.v.device
    )
    samples = torch.sum(mesh.v[mesh.f[f_idx]] * barycoords[..., None], axis=1)
    samples = samples.cpu().detach().numpy()

    with open(f"mesh/{obj}_eigfn{idx:03d}.npy", "wb") as f:
        np.save(f, samples.astype(np.float32))
        print(f"Saved mesh/{obj}_eigfn{idx:03d}.npy.")


def gen_maze_datapairs(n: int, replace: bool = False):
    if not replace and os.path.exists(f"mesh/maze_{n}x{n}.npz"):
        print(f"mesh/maze_{n}x{n}.npz exists. Skipping.")
        return

    v, f = igl.read_triangle_mesh(f"mesh/maze_{n}x{n}.obj")

    def sample_checkerboard(nsamples):
        x1 = np.random.rand(nsamples) * 4 - 2
        x2_ = np.random.rand(nsamples) - np.random.randint(0, 2, nsamples) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return 1.0 * np.concatenate([x1[:, None], x2[:, None]], 1) / 0.45

    def reject_outside_mesh(data, v_np, f_np):
        data = np.concatenate([data, np.zeros_like(data[:, 0:1])], axis=1)
        dist, _, _ = igl.signed_distance(data, v_np, f_np)
        data = data[dist == 0]
        return data

    np.random.seed(777)

    N = 100000

    x0 = np.random.randn(int(10 * N), 2)
    x0 = (x0 / 9.5 + 0.5) * n
    x0 = reject_outside_mesh(x0, v, f)
    Z0 = x0.shape[0] / (10 * N)
    x0 = x0[:N]

    x1 = sample_checkerboard(int(10 * N))
    x1 = (x1 / 9.5 + 0.5) * n
    x1 = reject_outside_mesh(x1, v, f)
    Z1 = x1.shape[0] / (10 * N)
    x1 = x1[:N]

    with open(f"mesh/maze_{n}x{n}.npz", "wb") as f:
        np.savez(f, x0=x0, x1=x1, Z0=Z0, Z1=Z1)


def gen_mazev2(n: int, replace: bool = False):
    filename = f"mesh/maze_{n}x{n}v2"

    if not replace and os.path.exists(f"{filename}.npz"):
        print(f"{filename}.npz exists. Skipping.")
        return

    v, f = igl.read_triangle_mesh(f"mesh/maze_{n}x{n}.obj")

    def reject_outside_mesh(data, v_np, f_np):
        data = np.concatenate([data, np.zeros_like(data[:, 0:1])], axis=1)
        dist, _, _ = igl.signed_distance(data, v_np, f_np)
        data = data[dist == 0]
        return data

    np.random.seed(777)
    torch.manual_seed(777)

    N = 100000

    std = 0.02 * 32 / (n * 10 + 2)

    if n % 2 == 0:
        offset = 1 / (n * 10 + 2) * 5 + 0.5
    else:
        offset = 0.5

    x0 = np.random.randn(int(10 * N), 2)
    x0 = (x0 * std + offset) * n
    x0 = reject_outside_mesh(x0, v, f)
    Z0 = x0.shape[0] / (10 * N)
    x0 = x0[:N]

    if n == 3:
        gmm = GaussianMM(
            [
                [1 / 6, 1 / 6],
                [5 / 6, 5 / 6],
            ],
            0.02,
        )
    elif n == 4:
        gmm = GaussianMM(
            [
                [1 / 8, 1 / 8],
                [1 / 8, 7 / 8],
                [7 / 8, 1 / 8],
            ],
            0.02 * 32 / (n * 10 + 2),
        )
    x1 = n * gmm.sample(N).detach().cpu().numpy()
    x1 = reject_outside_mesh(x1, v, f)
    Z1 = x1.shape[0] / (10 * N)
    x1 = x1[:N]

    with open(f"{filename}.npz", "wb") as f:
        np.savez(f, x0=x0, x1=x1, Z0=Z0, Z1=Z1, std=std)
        print(f"Saved {filename}.npz.")


def decimate_bunny():
    name = "bunny"
    v, _ = igl.read_triangle_mesh(f"mesh/{name}.obj")

    # this is a mesh that is closed.
    v_simp, f_simp, _ = igl.read_off(f"mesh/{name}.off")

    # align to original mesh.
    m1, s1 = v.mean(0), (v.max(0) - v.min(0))
    m2, s2 = v_simp.mean(0), (v_simp.max(0) - v_simp.min(0))
    v_simp = (v_simp - m2) / s2 * s1 + m1

    _, v_simp, f_simp, _, _ = igl.decimate(v_simp, f_simp, 5000)
    igl.write_obj(f"mesh/{name}_simp.obj", v_simp, f_simp)


def decimate_mesh(name: str, replace: bool = False):
    if not replace and os.path.exists(f"mesh/{name}_simp.obj"):
        print(f"mesh/{name}_simp.obj exists. Skipping.")
        return

    if name == "bunny":
        decimate_bunny()
        return

    v, f = igl.read_triangle_mesh(f"mesh/{name}.obj")
    _, v_simp, f_simp, _, _ = igl.decimate(v, f, 5000)
    igl.write_obj(f"mesh/{name}_simp.obj", v_simp, f_simp)


def pad2d(x):
    return np.pad(x, pad_width=((0, 0), (0, 1)))


if __name__ == "__main__":
    decimate_mesh("bunny")

    create_eigfn("bunny", 9, upsample=3, replace=False)
    create_eigfn("bunny", 49, upsample=3, replace=False)
    create_eigfn("bunny", 99, upsample=3, replace=False)

    create_eigfn("spot", 9, upsample=3, replace=False)
    create_eigfn("spot", 49, upsample=3, replace=False)
    create_eigfn("spot", 99, upsample=3, replace=False)

    gen_mazev2(3, replace=False)
    gen_mazev2(4, replace=False)
