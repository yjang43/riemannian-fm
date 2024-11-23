import h5py
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import make_interp_spline
import numpy as np


class RoboMimicDataset(Dataset):
    def __init__(
        self,
        hdf5_path,
        train=True,
        transform=None,
    ):
        hdf5_file = h5py.File(hdf5_path, 'r')

        mask = hdf5_file["mask/train"][:] if train else hdf5_file["mask/valid"]

        eps = []
        for ep_id in mask:
            eps.append(hdf5_file[f"data/{ep_id.decode('utf-8')}"])

        ep_lens = [ep["actions"].shape[0] for ep in eps]
        ep_bounds = [0]
        for ep_len in ep_lens:
            ep_bounds.append(ep_bounds[-1] + ep_len-1)

        eps_actions = []
        for ep in eps:
            eps_actions.append(self._interpolate_actions(ep))

        self.eps = eps
        self.ep_lens = ep_lens
        self.ep_bounds = ep_bounds
        self.eps_actions = eps_actions
        self.transform = transform

    def _binary_search(self, bounds, v):
        low, high = 0, len(bounds) - 1

        while low < high:
            mid = (low + high) // 2
            if bounds[mid] <= v < bounds[mid + 1]:
                return mid
            elif v < self.ep_bounds[mid]:
                high = mid - 1
            else:
                low = mid + 1

        return low

    def _interpolate_actions(self, ep):
        def make_zeroth_action(ep):
            # Repeat the first action.
            # Ideally, the current position of the end effector.
            return ep["actions"][0]

        actions = ep["actions"]
        N = actions.shape[0]
        zeroth_action = make_zeroth_action(ep)
        actions = np.concatenate([zeroth_action[None, :], actions])

        x = np.arange(0, N+1)
        y = make_interp_spline(x, actions)

        return y

    def _convert_idx(self, idx):
        ep_id = self._binary_search(self.ep_bounds, idx)
        n = idx - self.ep_bounds[ep_id]
        return ep_id, n

    def __len__(self):
        return self.ep_bounds[-1]

    def __getitem__(self, idx):
        ep_id, n = self._convert_idx(idx)
        N = self.ep_lens[ep_id]

        # For autoencoder.
        obs = self.eps[ep_id]["obs/agentview_image"][n]
        ni, nj = np.random.uniform(low=n, high=N, size=2)
        dist = np.abs(ni - nj)
        xi, xj = self.eps_actions[ep_id]([ni, nj])

        return obs, xi, xj, dist
    

