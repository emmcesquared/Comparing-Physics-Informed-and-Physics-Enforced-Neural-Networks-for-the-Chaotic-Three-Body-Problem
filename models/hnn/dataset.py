"""
PyTorch Dataset for HNN training on Three-Body Problem.

SAME FORMAT AS DNN/PINN/LNN:
- Input: [t, x2_0, z2_0]
- Target: [x1_t, z1_t, x2_t, z2_t]

For HNN physics loss, we also provide collocation points like PINN/LNN.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class HNNDataset(Dataset):
    """
    Dataset for HNN position prediction (SAME AS DNN/LNN).

    Each sample is (input, target) where:
    - input: [t, x2_0, z2_0] (time + initial body 2 position)
    - target: [x1_t, z1_t, x2_t, z2_t] (body 1 & 2 positions at time t)
    """

    def __init__(self, data_path, split='train', train_ratio=0.8, val_ratio=0.1,
                 seed=42, normalize=True):
        """
        Args:
            data_path: Path to .npz dataset file
            split: 'train', 'val', or 'test'
            train_ratio: Fraction of trajectories for training
            val_ratio: Fraction for validation (rest is test)
            seed: Random seed for split
            normalize: Whether to normalize inputs/outputs
        """
        data = np.load(data_path)
        self.t = data['t']      # (129,)
        self.Y = data['Y']      # (N_traj, 129, 18)
        self.y0 = data['y0']    # (N_traj, 18)

        n_traj = len(self.Y)
        n_time = len(self.t)

        # Split trajectories
        np.random.seed(seed)
        indices = np.random.permutation(n_traj)

        n_train = int(train_ratio * n_traj)
        n_val = int(val_ratio * n_traj)

        if split == 'train':
            traj_indices = indices[:n_train]
        elif split == 'val':
            traj_indices = indices[n_train:n_train + n_val]
        else:  # test
            traj_indices = indices[n_train + n_val:]

        self.traj_indices = traj_indices
        self.n_traj = len(traj_indices)
        self.n_time = n_time

        # Build flat arrays of (input, target) pairs
        self.inputs = []
        self.targets = []

        for traj_idx in traj_indices:
            x2_0 = self.y0[traj_idx, 3]  # x2 initial
            z2_0 = self.y0[traj_idx, 5]  # z2 initial

            for ti, t_val in enumerate(self.t):
                inp = np.array([t_val, x2_0, z2_0], dtype=np.float32)

                state = self.Y[traj_idx, ti]
                x1, z1 = state[0], state[2]
                x2, z2 = state[3], state[5]
                tgt = np.array([x1, z1, x2, z2], dtype=np.float32)

                self.inputs.append(inp)
                self.targets.append(tgt)

        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)

        # Store raw time and ICs for collocation sampling
        self.raw_t = self.t.copy()
        self.raw_ics = []
        for traj_idx in traj_indices:
            x2_0 = self.y0[traj_idx, 3]
            z2_0 = self.y0[traj_idx, 5]
            self.raw_ics.append([x2_0, z2_0])
        self.raw_ics = np.array(self.raw_ics, dtype=np.float32)

        # Normalization
        self.normalize = normalize
        if normalize:
            if split == 'train':
                self.input_mean = self.inputs.mean(axis=0)
                self.input_std = self.inputs.std(axis=0) + 1e-8
                self.target_mean = self.targets.mean(axis=0)
                self.target_std = self.targets.std(axis=0) + 1e-8
            else:
                self.input_mean = None
                self.input_std = None
                self.target_mean = None
                self.target_std = None

    def set_normalization(self, input_mean, input_std, target_mean, target_std):
        """Set normalization stats from training dataset."""
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = self.inputs[idx].copy()
        tgt = self.targets[idx].copy()

        if self.normalize and self.input_mean is not None:
            inp = (inp - self.input_mean) / self.input_std
            tgt = (tgt - self.target_mean) / self.target_std

        return torch.from_numpy(inp), torch.from_numpy(tgt)

    def get_raw_item(self, idx):
        """Get unnormalized item."""
        return self.inputs[idx], self.targets[idx]


class HNNCollocationDataset(Dataset):
    """
    Collocation points for HNN physics loss.

    Generates random (t, IC) pairs for computing Hamiltonian residual.
    """

    def __init__(self, train_dataset, n_collocation=10000, seed=42):
        """
        Args:
            train_dataset: HNNDataset (training set) for domain bounds
            n_collocation: Number of collocation points
            seed: Random seed
        """
        self.n_collocation = n_collocation
        self.seed = seed

        self.t_min = train_dataset.raw_t.min()
        self.t_max = train_dataset.raw_t.max()

        self.ic_min = train_dataset.raw_ics.min(axis=0) * 1.1
        self.ic_max = train_dataset.raw_ics.max(axis=0) * 1.1

        self._generate_points()

    def _generate_points(self):
        """Generate random collocation points."""
        np.random.seed(self.seed)

        t = np.random.uniform(self.t_min, self.t_max, (self.n_collocation, 1))

        ic = np.zeros((self.n_collocation, 2), dtype=np.float32)
        for i in range(2):
            ic[:, i] = np.random.uniform(self.ic_min[i], self.ic_max[i],
                                          self.n_collocation)

        self.t_coll = t.astype(np.float32)
        self.ic_coll = ic.astype(np.float32)

    def __len__(self):
        return self.n_collocation

    def __getitem__(self, idx):
        return (torch.from_numpy(self.t_coll[idx]),
                torch.from_numpy(self.ic_coll[idx]))


class HNNTrajectoryDataset(Dataset):
    """
    Dataset returning full trajectories for HNN evaluation.
    SAME AS DNN/LNN's TrajectoryDataset.
    """

    def __init__(self, data_path, split='test', train_ratio=0.8, val_ratio=0.1, seed=42):
        data = np.load(data_path)
        self.t = data['t']
        self.Y = data['Y']
        self.y0 = data['y0']

        n_traj = len(self.Y)

        np.random.seed(seed)
        indices = np.random.permutation(n_traj)

        n_train = int(train_ratio * n_traj)
        n_val = int(val_ratio * n_traj)

        if split == 'train':
            self.traj_indices = indices[:n_train]
        elif split == 'val':
            self.traj_indices = indices[n_train:n_train + n_val]
        else:
            self.traj_indices = indices[n_train + n_val:]

    def __len__(self):
        return len(self.traj_indices)

    def __getitem__(self, idx):
        traj_idx = self.traj_indices[idx]

        x2_0 = self.y0[traj_idx, 3]
        z2_0 = self.y0[traj_idx, 5]
        ic = np.array([x2_0, z2_0], dtype=np.float32)

        traj = self.Y[traj_idx]

        # Positions (all 3 bodies, x-z only)
        positions = np.zeros((len(self.t), 6), dtype=np.float32)
        positions[:, 0] = traj[:, 0]   # x1
        positions[:, 1] = traj[:, 2]   # z1
        positions[:, 2] = traj[:, 3]   # x2
        positions[:, 3] = traj[:, 5]   # z2
        positions[:, 4] = traj[:, 6]   # x3
        positions[:, 5] = traj[:, 8]   # z3

        # Velocities
        velocities = np.zeros((len(self.t), 6), dtype=np.float32)
        velocities[:, 0] = traj[:, 9]    # vx1
        velocities[:, 1] = traj[:, 11]   # vz1
        velocities[:, 2] = traj[:, 12]   # vx2
        velocities[:, 3] = traj[:, 14]   # vz2
        velocities[:, 4] = traj[:, 15]   # vx3
        velocities[:, 5] = traj[:, 17]   # vz3

        return {
            'ic': torch.from_numpy(ic),
            't': torch.from_numpy(self.t.astype(np.float32)),
            'positions': torch.from_numpy(positions),
            'velocities': torch.from_numpy(velocities),
            'traj_idx': traj_idx
        }


def create_hnn_dataloaders(data_path, batch_size=2048, n_collocation=10000, num_workers=0):
    """
    Create train/val/test dataloaders for HNN.

    Returns:
        train_loader, val_loader, test_loader, collocation_loader, norm_stats
    """
    train_ds = HNNDataset(data_path, split='train', normalize=True)
    val_ds = HNNDataset(data_path, split='val', normalize=True)
    test_ds = HNNDataset(data_path, split='test', normalize=True)

    # Share normalization
    val_ds.set_normalization(
        train_ds.input_mean, train_ds.input_std,
        train_ds.target_mean, train_ds.target_std
    )
    test_ds.set_normalization(
        train_ds.input_mean, train_ds.input_std,
        train_ds.target_mean, train_ds.target_std
    )

    # Collocation dataset
    collocation_ds = HNNCollocationDataset(train_ds, n_collocation=n_collocation)

    norm_stats = {
        'input_mean': train_ds.input_mean,
        'input_std': train_ds.input_std,
        'target_mean': train_ds.target_mean,
        'target_std': train_ds.target_std
    }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    collocation_loader = DataLoader(collocation_ds, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader, collocation_loader, norm_stats


if __name__ == "__main__":
    data_path = "../../dataset generation/pinn_three_body_dataset.npz"

    print("Testing HNNDataset (DNN-compatible)...")
    train_ds = HNNDataset(data_path, split='train')
    val_ds = HNNDataset(data_path, split='val')
    test_ds = HNNDataset(data_path, split='test')

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    inp, tgt = train_ds[0]
    print(f"\nSample input shape: {inp.shape}")   # [3]
    print(f"Sample target shape: {tgt.shape}")     # [4]

    print("\n" + "="*50)
    print("Testing HNNCollocationDataset...")
    coll_ds = HNNCollocationDataset(train_ds, n_collocation=1000)
    print(f"Collocation points: {len(coll_ds)}")

    t_coll, ic_coll = coll_ds[0]
    print(f"t shape: {t_coll.shape}, ic shape: {ic_coll.shape}")

    print("\n" + "="*50)
    print("Testing HNNTrajectoryDataset...")
    traj_ds = HNNTrajectoryDataset(data_path, split='test')
    print(f"Test trajectories: {len(traj_ds)}")

    sample = traj_ds[0]
    print(f"IC: {sample['ic'].shape}")
    print(f"Positions: {sample['positions'].shape}")
