import os
import math

from tqdm import tqdm
import numpy as np
import torch
import hydra

from manifm.eval_utils import load_model
from manifm.datasets import get_loaders, get_manifold


@hydra.main(version_base=None, config_path="configs", config_name="reflow")
def main(cfg):
    ckpt_path = cfg.get("ckpt", None)

    ckpt_cfg, pl_model = load_model(ckpt_path)
    manifold, dim = get_manifold(ckpt_cfg)

    # Config.
    device = cfg.get("device", "cuda")
    data_size = cfg.get("data_size", None)
    batch_size = cfg.get("batch_size", None)
    num_steps = cfg.get("num_steps", 1000)
    datadir = cfg.get("datadir", None)
    data = ckpt_cfg.get("data", None)

    pl_model.to(device)
    pl_model.train(mode=False)
    os.makedirs(datadir, exist_ok=True)

    x0 = np.empty((data_size, dim), dtype=np.float32)
    x1 = np.empty((data_size, dim), dtype=np.float32)

    for i in tqdm(range(math.ceil(data_size / batch_size))):
        N = min(batch_size, data_size - i * batch_size)
        x0_ = manifold.random_base(N, dim)
        x1_ = pl_model.sample(None, device, x0_.to(device), num_steps=num_steps).cpu()

        x0[batch_size*i: batch_size*i + N] = x0_
        x1[batch_size*i: batch_size*i + N] = x1_

    np.savez(f"{datadir}/reflow_{data}.npz", x0=x0, x1=x1)


if __name__ == "__main__":
    main()