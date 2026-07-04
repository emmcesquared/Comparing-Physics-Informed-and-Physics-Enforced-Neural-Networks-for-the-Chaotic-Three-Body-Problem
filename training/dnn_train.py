"""
Training script for Three-Body DNN.

Architecture: 10 hidden layers, 128 neurons, ReLU activation
Optimizer: Adam
Loss: MAE (Mean Absolute Error)
Epochs: 10,000
Batch size: 5,000
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

from model import ThreeBodyDNN, ThreeBodyDNNLarge, ThreeBodyDNNSmall
from dataset import create_dataloaders


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, loader, criterion, device):
    """Validate the model."""
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
    epochs=10000,
    batch_size=5000,
    lr=1e-3,
    weight_decay=0,
    hidden_dim=128,
    num_layers=10,
    dropout=0.0,
    scheduler_type='plateau',
    patience=500,
    save_every=1000,
    device='cuda'
):
    """
    Main training function.

    Args:
        data_path: Path to dataset .npz file
        output_dir: Directory to save checkpoints and logs
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: L2 regularization
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        dropout: Dropout rate
        scheduler_type: 'plateau' or 'cosine'
        patience: Early stopping patience
        save_every: Save checkpoint every N epochs
        device: 'cuda' or 'cpu'
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print(f"\nLoading data from {data_path}...")
    train_loader, val_loader, test_loader, norm_stats = create_dataloaders(
        data_path, batch_size=batch_size
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Model
    model = ThreeBodyDNN(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    model = model.to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Loss & Optimizer
    criterion = nn.L1Loss()  # MAE
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                       patience=100, verbose=True)
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
        'num_layers': num_layers,
        'dropout': dropout,
        'scheduler_type': scheduler_type,
        'device': str(device),
        'n_params': model.count_parameters(),
        'norm_stats': {k: v.tolist() for k, v in norm_stats.items()}
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
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
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
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
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Final test loss: {test_loss:.6f}")
    print(f"Model saved to: {output_dir}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Three-Body DNN')
    parser.add_argument('--data', type=str,
                        default='../../dataset generation/pinn_three_body_dataset.npz',
                        help='Path to dataset')
    parser.add_argument('--output', type=str, default='./checkpoints',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.0)
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
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        patience=args.patience,
        device=args.device
    )


if __name__ == '__main__':
    main()
