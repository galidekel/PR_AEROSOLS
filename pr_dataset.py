import numpy as np
import torch
from torch.utils.data import Dataset


class PRAfricaDustRouteDataset(Dataset):
    def __init__(
        self,
        X_dust,
        X_route,
        Y,
        aeronet_t=None,
        dust_mean=None,
        dust_std=None,
        route_mean=None,
        route_std=None,
        y_mean=None,
        y_std=None,
        normalize_x=True,
        normalize_y=False,
    ):
        """
        X_dust:  [N, T, 2, H_d, W_d]
        X_route: [N, T, C_r, H_r, W_r]
        Y:       [N, L]
        aeronet_t: optional [N, A]

        Normalization:
          dust_mean/std  shape [1,1,2,1,1] or broadcastable
          route_mean/std shape [1,1,C_r,1,1] or broadcastable
          y_mean/std     shape [1,L] or broadcastable
        """
        self.X_dust = X_dust
        self.X_route = X_route
        self.Y = Y
        self.aeronet_t = aeronet_t

        self.dust_mean = dust_mean
        self.dust_std = dust_std
        self.route_mean = route_mean
        self.route_std = route_std
        self.y_mean = y_mean
        self.y_std = y_std

        self.normalize_x = normalize_x
        self.normalize_y = normalize_y

        n = len(Y)
        if len(X_dust) != n or len(X_route) != n:
            raise ValueError("X_dust, X_route, and Y must have same length")
        if aeronet_t is not None and len(aeronet_t) != n:
            raise ValueError("aeronet_t must have same length as Y")

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x_dust = self.X_dust[idx].astype(np.float32)    # [T,2,H_d,W_d]
        x_route = self.X_route[idx].astype(np.float32)  # [T,C_r,H_r,W_r]
        y = self.Y[idx].astype(np.float32)              # [L]

        if self.normalize_x:
            if self.dust_mean is not None and self.dust_std is not None:
                x_dust = (x_dust - self.dust_mean) / (self.dust_std + 1e-8)

            if self.route_mean is not None and self.route_std is not None:
                x_route = (x_route - self.route_mean) / (self.route_std + 1e-8)

        if self.normalize_y and self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / (self.y_std + 1e-8)

        sample = {
            "x_dust": torch.from_numpy(x_dust),     # [T,2,H_d,W_d]
            "x_route": torch.from_numpy(x_route),   # [T,C_r,H_r,W_r]
            "y": torch.from_numpy(y),               # [L]
        }

        if self.aeronet_t is not None:
            sample["aeronet_t"] = torch.from_numpy(self.aeronet_t[idx].astype(np.float32))

        return sample
