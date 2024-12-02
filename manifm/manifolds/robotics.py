
import math
import numpy as np
from geoopt import Manifold
import torch
import scipy.linalg
from torch.func import jacrev, vmap
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Robotics:
    name = "Robotics"
    ndim = 1

    def __init__(self):
        super().__init__()

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def dist(self, x: torch.Tensor, y: torch.Tensor):
        return (x - y).norm(dim=-1)
