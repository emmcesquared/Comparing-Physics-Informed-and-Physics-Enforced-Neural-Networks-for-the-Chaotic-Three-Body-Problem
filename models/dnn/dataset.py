"""
PyTorch Dataset for Three-Body Problem.

Converts the raw trajectory data into (input, target) pairs for DNN training.

Input format: [t, x2_0, z2_0]
Target format: [x1_t, z1_t, x2_t, z2_t]
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ThreeBodyDataset(Dataset):
    """
    Dataset for three-body position prediction.

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
        # Load data
        data = np.load(data_path)
        self.t = data['t']  # (129,)
        self.Y = data['Y']  # (N_traj, 129, 18)
        self.y0 = data['y0']  # (N_traj, 18)

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

        # Extract relevant data for this split
        self.traj_indices = traj_indices
        self.n_traj = len(traj_indices)
        self.n_time = n_time

        # Build flat arrays of (input, target) pairs
        # Each trajectory contributes n_time samples
        self.inputs = []
        self.targets = []

        for traj_idx in traj_indices:
            # Initial condition: body 2 position (x2_0, z2_0)
            x2_0 = self.y0[traj_idx, 3]  # x2 initial
            z2_0 = self.y0[traj_idx, 5]  # z2 initial (y-coord is always 0)

            for ti, t_val in enumerate(self.t):
                # Input: [t, x2_0, z2_0]
                inp = np.array([t_val, x2_0, z2_0], dtype=np.float32)

                # Target: [x1, z1, x2, z2] at time t
                state = self.Y[traj_idx, ti]
                x1, z1 = state[0], state[2]  # body 1: indices 0,1,2 -> x,y,z
                x2, z2 = state[3], state[5]  # body 2: indices 3,4,5 -> x,y,z
                tgt = np.array([x1, z1, x2, z2], dtype=np.float32)

                self.inputs.append(inp)
                self.targets.append(tgt)

        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)

        # Normalization statistics (computed on training data)
        self.normalize = normalize
        if normalize:
            if split == 'train':
                self.input_mean = self.inputs.mean(axis=0)
                self.input_std = self.inputs.std(axis=0) + 1e-8
                self.target_mean = self.targets.mean(axis=0)
                self.target_std = self.targets.std(axis=0) + 1e-8
            else:
                # Will be set from training dataset
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


class ThreeBodyTrajectoryDataset(Dataset):
    """
    Dataset that returns full trajectories (for evaluation/rollout).

    Each sample is a full trajectory with initial conditions.
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

        # Initial conditions for body 2
        x2_0 = self.y0[traj_idx, 3]
        z2_0 = self.y0[traj_idx, 5]
        ic = np.array([x2_0, z2_0], dtype=np.float32)

        # Full trajectory positions (all 3 bodies, x-z only)
        traj = self.Y[traj_idx]  # (129, 18)

        # Extract x,z for all bodies: shape (129, 6)
        positions = np.zeros((len(self.t), 6), dtype=np.float32)
        positions[:, 0] = traj[:, 0]  # x1
        positions[:, 1] = traj[:, 2]  # z1
        positions[:, 2] = traj[:, 3]  # x2
        positions[:, 3] = traj[:, 5]  # z2
        positions[:, 4] = traj[:, 6]  # x3
        positions[:, 5] = traj[:, 8]  # z3

        # Velocities for energy computation
        velocities = np.zeros((len(self.t), 6), dtype=np.float32)
        velocities[:, 0] = traj[:, 9]   # vx1
        velocities[:, 1] = traj[:, 11]  # vz1
        velocities[:, 2] = traj[:, 12]  # vx2
        velocities[:, 3] = traj[:, 14]  # vz2
        velocities[:, 4] = traj[:, 15]  # vx3
        velocities[:, 5] = traj[:, 17]  # vz3

        return {
            'ic': torch.from_numpy(ic),
            't': torch.from_numpy(self.t.astype(np.float32)),
            'positions': torch.from_numpy(positions),
            'velocities': torch.from_numpy(velocities),
            'traj_idx': traj_idx
        }


def create_dataloaders(data_path, batch_size=5000, num_workers=0):
    """
    Create train/val/test dataloaders.

    Returns:
        train_loader, val_loader, test_loader, norm_stats
    """
    train_ds = ThreeBodyDataset(data_path, split='train', normalize=True)
    val_ds = ThreeBodyDataset(data_path, split='val', normalize=True)
    test_ds = ThreeBodyDataset(data_path, split='test', normalize=True)

    # Share normalization stats
    val_ds.set_normalization(
        train_ds.input_mean, train_ds.input_std,
        train_ds.target_mean, train_ds.target_std
    )
    test_ds.set_normalization(
        train_ds.input_mean, train_ds.input_std,
        train_ds.target_mean, train_ds.target_std
    )

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

    return train_loader, val_loader, test_loader, norm_stats


if __name__ == "__main__":
    # Test dataset
    data_path = "../../dataset generation/pinn_three_body_dataset.npz"

    print("Testing ThreeBodyDataset...")
    train_ds = ThreeBodyDataset(data_path, split='train')
    val_ds = ThreeBodyDataset(data_path, split='val')
    test_ds = ThreeBodyDataset(data_path, split='test')

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    inp, tgt = train_ds[0]
    print(f"\nSample input shape: {inp.shape}")
    print(f"Sample target shape: {tgt.shape}")

    print("\n" + "="*50)
    print("Testing ThreeBodyTrajectoryDataset...")
    traj_ds = ThreeBodyTrajectoryDataset(data_path, split='test')
    print(f"Test trajectories: {len(traj_ds)}")

    sample = traj_ds[0]
    print(f"IC shape: {sample['ic'].shape}")
    print(f"Time shape: {sample['t'].shape}")
    print(f"Positions shape: {sample['positions'].shape}")
