"""
Training script for Three-Body PINN with ResNet.

Architecture: ResNet with skip connections
Loss: L_total = L_data + λ * L_physics
- L_data: MAE on position predictions
- L_physics: ODE residual via autodiff

Optimizer: Adam
Epochs: 2000
Batch size: 2048
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from model import ThreeBodyPINN, gravitational_acceleration
from dataset import create_pinn_dataloaders


def compute_physics_loss(model, t, ic, norm_stats, device):
    """
    Compute physics loss (ODE residual) via automatic differentiation.

    L_physics = |q̈_pred - F(q_pred)|

    Args:
        model: PINN model
        t: (batch, 1) time values (unnormalized)
        ic: (batch, 2) initial conditions [x2_0, z2_0] (unnormalized)
        norm_stats: normalization statistics
        device: torch device

    Returns:
        physics_loss: scalar tensor
    """
    batch_size = t.shape[0]

    # Move to device and enable gradients on time
    t = t.to(device).requires_grad_(True)
    ic = ic.to(device)

    # Get normalization constants
    input_mean = torch.tensor(norm_stats['input_mean'], device=device, dtype=torch.float32)
    input_std = torch.tensor(norm_stats['input_std'], device=device, dtype=torch.float32)
    target_mean = torch.tensor(norm_stats['target_mean'], device=device, dtype=torch.float32)
    target_std = torch.tensor(norm_stats['target_std'], device=device, dtype=torch.float32)

    # Build input tensor using torch.cat (gradient-friendly)
    inputs = torch.cat([t, ic], dim=1)  # (batch, 3)

    # Normalize inputs
    inputs_norm = (inputs - input_mean) / input_std

    # Forward pass
    outputs_norm = model(inputs_norm)  # (batch, 4)

    # Denormalize outputs
    outputs = outputs_norm * target_std + target_mean  # [x1, z1, x2, z2]

    # Compute velocities (d(position)/d(time)) for each output dimension
    # The grad is computed w.r.t the original unnormalized time
    ones = torch.ones_like(outputs[:, 0])
    
    vel_list = []
    for i in range(4):
        grad_i = torch.autograd.grad(
            outputs[:, i], t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True
        )[0]
        vel_list.append(grad_i)
    
    vel = torch.cat(vel_list, dim=1)  # (batch, 4)

    # Compute accelerations (d²(position)/d(time²))
    acc_list = []
    for i in range(4):
        grad_i = torch.autograd.grad(
            vel[:, i], t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True
        )[0]
        acc_list.append(grad_i)
    
    acc_pred = torch.cat(acc_list, dim=1)  # (batch, 4)

    # Compute expected acceleration from gravitational law
    acc_true = gravitational_acceleration(outputs)  # (batch, 4)

    # Physics loss: MAE of acceleration residual
    physics_loss = torch.mean(torch.abs(acc_pred - acc_true))

    return physics_loss


def train_epoch(model, data_loader, collocation_loader, optimizer,
                criterion, norm_stats, device, lambda_physics=1.0):
    """
    Train for one epoch with combined data + physics loss.

    Args:
        model: PINN model
        data_loader: DataLoader for supervised data
        collocation_loader: DataLoader for collocation points
        optimizer: optimizer
        criterion: loss function for data loss
        norm_stats: normalization statistics
        device: torch device
        lambda_physics: weight for physics loss

    Returns:
        avg_loss, avg_data_loss, avg_physics_loss
    """
    model.train()
    total_loss = 0
    total_data_loss = 0
    total_physics_loss = 0
    n_batches = 0

    collocation_iter = iter(collocation_loader)

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Data loss
        outputs = model(inputs)
        data_loss = criterion(outputs, targets)

        # Physics loss (from collocation points)
        try:
            t_coll, ic_coll = next(collocation_iter)
        except StopIteration:
            collocation_iter = iter(collocation_loader)
            t_coll, ic_coll = next(collocation_iter)

        physics_loss = compute_physics_loss(model, t_coll, ic_coll, norm_stats, device)

        # Combined loss
        loss = data_loss + lambda_physics * physics_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_data_loss += data_loss.item()
        total_physics_loss += physics_loss.item()
        n_batches += 1

    return total_loss / n_batches, total_data_loss / n_batches, total_physics_loss / n_batches


def validate(model, loader, criterion, device):
    """Validate the model (data loss only)."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train(
    data_path,
    output_dir,
    epochs=2000,
    batch_size=2048,
    lr=1e-3,
    weight_decay=1e-5,
    hidden_dim=128,
    num_blocks=5,
    lambda_physics=1.0,
    n_collocation=10000,
    scheduler_type='plateau',
    patience=250,
    save_every=500,
    device='cuda'
):
    """
    Main PINN training function.
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print(f"\nLoading data from {data_path}...")
    train_loader, val_loader, test_loader, collocation_loader, norm_stats = \
        create_pinn_dataloaders(data_path, batch_size=batch_size, n_collocation=n_collocation)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Collocation batches: {len(collocation_loader)}")

    # Model
    model = ThreeBodyPINN(hidden_dim=hidden_dim, num_blocks=num_blocks)
    model = model.to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Loss & Optimizer
    criterion = nn.L1Loss()  # MAE
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training config
    config = {
        'data_path': str(data_path),
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'hidden_dim': hidden_dim,
        'num_blocks': num_blocks,
        'lambda_physics': lambda_physics,
        'n_collocation': n_collocation,
        'scheduler_type': scheduler_type,
        'device': str(device),
        'n_params': model.count_parameters(),
        'norm_stats': {k: v.tolist() for k, v in norm_stats.items()}
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting PINN training (λ_physics = {lambda_physics})...")
    print(f"{'='*70}\n")

    history = {
        'train_loss': [], 'val_loss': [],
        'data_loss': [], 'physics_loss': [], 'lr': []
    }
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, data_loss, physics_loss = train_epoch(
            model, train_loader, collocation_loader, optimizer,
            criterion, norm_stats, device, lambda_physics
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['data_loss'].append(data_loss)
        history['physics_loss'].append(physics_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Scheduler step
        if scheduler_type == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Logging
        if epoch % 100 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:5d}/{epochs} | "
                  f"Loss: {train_loss:.4f} (D:{data_loss:.4f} P:{physics_loss:.4f}) | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'norm_stats': norm_stats
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'norm_stats': norm_stats
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'norm_stats': norm_stats
    }, output_dir / 'final_model.pt')

    # Save history
    np.savez(output_dir / 'history.npz', **history)

    # Final test evaluation
    test_loss = validate(model, test_loader, criterion, device)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("PINN Training Complete!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Final test loss: {test_loss:.6f}")
    print(f"Model saved to: {output_dir}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Three-Body PINN')
    parser.add_argument('--data', type=str,
                        default='../../dataset generation/pinn_three_body_dataset.npz',
                        help='Path to dataset')
    parser.add_argument('--output', type=str, default='./checkpoints',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-blocks', type=int, default=5)
    parser.add_argument('--lambda-physics', type=float, default=1.0,
                        help='Weight for physics loss')
    parser.add_argument('--n-collocation', type=int, default=10000)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'])
    parser.add_argument('--patience', type=int, default=250)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        lambda_physics=args.lambda_physics,
        n_collocation=args.n_collocation,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        patience=args.patience,
        device=args.device
    )


if __name__ == '__main__':
    main()
