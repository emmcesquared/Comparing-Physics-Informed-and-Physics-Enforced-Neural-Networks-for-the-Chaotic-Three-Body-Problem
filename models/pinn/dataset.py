"""
PyTorch Dataset for PINN Three-Body Problem.

Extends the DNN dataset with:
1. Same (input, target) pairs for data loss
2. Collocation point sampling for physics residual loss
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import importlib.util

# Load DNN dataset module explicitly to avoid name collision
_current_dir = os.path.dirname(os.path.abspath(__file__))
_dnn_dataset_path = os.path.join(os.path.dirname(_current_dir), 'dnn', 'dataset.py')
_spec = importlib.util.spec_from_file_location("dnn_dataset", _dnn_dataset_path)
_dnn_dataset_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dnn_dataset_module)

ThreeBodyDataset = _dnn_dataset_module.ThreeBodyDataset
ThreeBodyTrajectoryDataset = _dnn_dataset_module.ThreeBodyTrajectoryDataset


class PINNDataset(ThreeBodyDataset):
    """
    Dataset for PINN training with both data and physics loss.

    Inherits from ThreeBodyDataset and adds collocation point handling.
    """

    def __init__(self, data_path, split='train', train_ratio=0.8, val_ratio=0.1,
                 seed=42, normalize=True):
        super().__init__(data_path, split, train_ratio, val_ratio, seed, normalize)

        # Store raw time and IC data for collocation points
        self.t_raw = self.inputs[:, 0].copy()  # All time values
        self.ic_raw = self.inputs[:, 1:].copy()  # All ICs [x2_0, z2_0]


class PINNCollocationDataset(Dataset):
    """
    Dataset for sampling collocation points for physics loss.

    Samples random (t, IC) pairs within the training domain.
    """

    def __init__(self, data_path, n_collocation=10000, seed=42):
        """
        Args:
            data_path: Path to dataset
            n_collocation: Number of collocation points per epoch
            seed: Random seed
        """
        data = np.load(data_path)
        self.t_range = (data['t'].min(), data['t'].max())
        self.y0 = data['y0']  # (N_traj, 18)

        # Extract IC range [x2_0, z2_0]
        self.ic_data = np.stack([
            self.y0[:, 3],  # x2_0
            self.y0[:, 5]   # z2_0
        ], axis=1)

        self.ic_min = self.ic_data.min(axis=0)
        self.ic_max = self.ic_data.max(axis=0)

        self.n_collocation = n_collocation
        self.seed = seed
        self._resample()

    def _resample(self):
        """Resample collocation points."""
        np.random.seed(self.seed)

        # Random times
        t = np.random.uniform(self.t_range[0], self.t_range[1], self.n_collocation)

        # Random ICs within data range (with small buffer)
        buffer = 0.1 * (self.ic_max - self.ic_min)
        ic = np.random.uniform(
            self.ic_min - buffer,
            self.ic_max + buffer,
            (self.n_collocation, 2)
        )

        self.collocation_t = t.astype(np.float32)
        self.collocation_ic = ic.astype(np.float32)

    def __len__(self):
        return self.n_collocation

    def __getitem__(self, idx):
        t = torch.tensor([self.collocation_t[idx]], dtype=torch.float32)
        ic = torch.from_numpy(self.collocation_ic[idx])
        return t, ic


def create_pinn_dataloaders(data_path, batch_size=2048, n_collocation=10000, num_workers=0):
    """
    Create train/val/test dataloaders for PINN.

    Returns:
        train_loader: DataLoader for supervised data loss
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing
        collocation_loader: DataLoader for physics loss
        norm_stats: Normalization statistics
    """
    train_ds = PINNDataset(data_path, split='train', normalize=True)
    val_ds = PINNDataset(data_path, split='val', normalize=True)
    test_ds = PINNDataset(data_path, split='test', normalize=True)

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

    # Collocation dataset for physics loss
    collocation_ds = PINNCollocationDataset(data_path, n_collocation=n_collocation)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    collocation_loader = DataLoader(collocation_ds, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, collocation_loader, norm_stats


if __name__ == "__main__":
    # Test dataset
    data_path = "../../dataset generation/pinn_three_body_dataset.npz"

    print("Testing PINNDataset...")
    train_ds = PINNDataset(data_path, split='train')
    print(f"Train samples: {len(train_ds)}")

    inp, tgt = train_ds[0]
    print(f"Sample input shape: {inp.shape}")
    print(f"Sample target shape: {tgt.shape}")

    print("\n" + "="*50)
    print("Testing PINNCollocationDataset...")
    collocation_ds = PINNCollocationDataset(data_path, n_collocation=1000)
    print(f"Collocation points: {len(collocation_ds)}")

    t, ic = collocation_ds[0]
    print(f"t shape: {t.shape}")
    print(f"IC shape: {ic.shape}")

    print("\n" + "="*50)
    print("Testing create_pinn_dataloaders...")
    train_loader, val_loader, test_loader, collocation_loader, norm_stats = \
        create_pinn_dataloaders(data_path, batch_size=512, n_collocation=1000)
    print(f"Train batches: {len(train_loader)}")
    print(f"Collocation batches: {len(collocation_loader)}")
