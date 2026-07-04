"""
Training script for Hamiltonian Neural Network (HNN).

SAME INTERFACE AS DNN/PINN/LNN - predicts positions directly.

Loss function:
    L_total = L_data + lambda_physics * L_physics + lambda_hamilton * L_hamilton + lambda_ham_sup * L_ham_sup

Where:
- L_data: Position prediction loss (MAE, same as DNN)
- L_physics: ODE residual via autodiff (same as PINN/LNN)
- L_hamilton: Hamilton's equations residual (HNN-specific)
    R_H = ||J * grad(H) - s_dot||  where J is the symplectic matrix
- L_ham_sup: Hamiltonian supervision (H_learned vs H_true = T + V)

The HNN combines:
1. Direct position prediction (like DNN)
2. Physics constraint via ODE residual (like PINN)
3. Hamiltonian structure constraint (unique to HNN)
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

from model import ThreeBodyHNN, gravitational_acceleration
from dataset import create_hnn_dataloaders


def compute_physics_loss(model, t, ic, norm_stats, device):
    """
    Compute physics loss (ODE residual) via automatic differentiation.
    SAME AS PINN/LNN - ensures predicted positions satisfy Newton's laws.

    L_physics = |q_ddot_pred - F(q_pred)|

    Args:
        model: HNN model
        t: (batch, 1) time values (unnormalized)
        ic: (batch, 2) initial conditions [x2_0, z2_0] (unnormalized)
        norm_stats: normalization statistics
        device: torch device

    Returns:
        physics_loss: scalar tensor
    """
    batch_size = t.shape[0]

    t = t.to(device).requires_grad_(True)
    ic = ic.to(device)

    input_mean = torch.tensor(norm_stats['input_mean'], device=device, dtype=torch.float32)
    input_std = torch.tensor(norm_stats['input_std'], device=device, dtype=torch.float32)
    target_mean = torch.tensor(norm_stats['target_mean'], device=device, dtype=torch.float32)
    target_std = torch.tensor(norm_stats['target_std'], device=device, dtype=torch.float32)

    inputs = torch.cat([t, ic], dim=1)
    inputs_norm = (inputs - input_mean) / input_std

    outputs_norm = model(inputs_norm)
    outputs = outputs_norm * target_std + target_mean  # [x1, z1, x2, z2]

    # Compute velocities (dq/dt)
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
    vel = torch.cat(vel_list, dim=1)

    # Compute accelerations (d2q/dt2)
    acc_list = []
    for i in range(4):
        grad_i = torch.autograd.grad(
            vel[:, i], t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True
        )[0]
        acc_list.append(grad_i)
    acc_pred = torch.cat(acc_list, dim=1)

    acc_true = gravitational_acceleration(outputs)

    physics_loss = torch.mean(torch.abs(acc_pred - acc_true))

    return physics_loss


def compute_hamilton_loss(model, t, ic, norm_stats, device):
    """
    Compute Hamilton's equations residual.

    Hamilton's equations:
        dq/dt =  dH/dp
        dp/dt = -dH/dq

    For m=1: p = v, so dp/dt = dv/dt = acceleration.

    The residual measures how well the learned Hamiltonian generates the
    correct equations of motion:
        R_H = ||dH/dp - dq/dt||  +  ||-dH/dq - dp/dt||

    Args:
        model: HNN model with hamiltonian and compute_hamiltonian_vector_field_residual
        t: (batch, 1) time values
        ic: (batch, 2) initial conditions
        norm_stats: normalization statistics
        device: torch device

    Returns:
        hamilton_loss: scalar tensor
    """
    batch_size = t.shape[0]

    t = t.to(device).requires_grad_(True)
    ic = ic.to(device)

    input_mean = torch.tensor(norm_stats['input_mean'], device=device, dtype=torch.float32)
    input_std = torch.tensor(norm_stats['input_std'], device=device, dtype=torch.float32)
    target_mean = torch.tensor(norm_stats['target_mean'], device=device, dtype=torch.float32)
    target_std = torch.tensor(norm_stats['target_std'], device=device, dtype=torch.float32)

    inputs = torch.cat([t, ic], dim=1)
    inputs_norm = (inputs - input_mean) / input_std

    outputs_norm = model(inputs_norm)
    q = outputs_norm * target_std + target_mean  # positions [x1, z1, x2, z2]

    # Compute velocities via autodiff: dq/dt
    ones = torch.ones_like(q[:, 0])
    vel_list = []
    for i in range(4):
        grad_i = torch.autograd.grad(
            q[:, i], t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True
        )[0]
        vel_list.append(grad_i)
    qdot = torch.cat(vel_list, dim=1)  # velocities = momenta (m=1)

    # Compute accelerations via autodiff: d2q/dt2 = dp/dt (since m=1)
    acc_list = []
    for i in range(4):
        grad_i = torch.autograd.grad(
            qdot[:, i], t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True
        )[0]
        acc_list.append(grad_i)
    pdot = torch.cat(acc_list, dim=1)  # dp/dt = acceleration (m=1)

    # Compute Hamiltonian vector field: dq/dt_H = dH/dp, dp/dt_H = -dH/dq
    q_ham = q.clone().requires_grad_(True)
    p_ham = qdot.clone().requires_grad_(True)  # p = mv = v (m=1)

    dq_dt_H, dp_dt_H = model.compute_hamiltonian_vector_field_residual(q_ham, p_ham)

    # Hamilton's equations residual:
    # ||dH/dp - dq/dt|| + ||-dH/dq - dp/dt||
    residual_q = torch.mean(torch.abs(dq_dt_H - qdot))   # dq/dt = dH/dp
    residual_p = torch.mean(torch.abs(dp_dt_H - pdot))    # dp/dt = -dH/dq

    hamilton_loss = residual_q + residual_p

    return hamilton_loss


def compute_hamiltonian_supervision_loss(model, t, ic, norm_stats, device, G=1.0):
    """
    Compute Hamiltonian supervision loss.

    Trains the Hamiltonian network to predict the true Hamiltonian H = T + V,
    where T is kinetic energy and V is gravitational potential energy.

    Args:
        model: HNN model with hamiltonian method
        t: (batch, 1) time values
        ic: (batch, 2) initial conditions
        norm_stats: normalization statistics
        device: torch device
        G: gravitational constant

    Returns:
        ham_loss: scalar tensor
    """
    batch_size = t.shape[0]

    t = t.to(device).requires_grad_(True)
    ic = ic.to(device)

    input_mean = torch.tensor(norm_stats['input_mean'], device=device, dtype=torch.float32)
    input_std = torch.tensor(norm_stats['input_std'], device=device, dtype=torch.float32)
    target_mean = torch.tensor(norm_stats['target_mean'], device=device, dtype=torch.float32)
    target_std = torch.tensor(norm_stats['target_std'], device=device, dtype=torch.float32)

    inputs = torch.cat([t, ic], dim=1)
    inputs_norm = (inputs - input_mean) / input_std

    outputs_norm = model(inputs_norm)
    q = outputs_norm * target_std + target_mean  # positions

    # Compute velocities via autodiff
    ones = torch.ones_like(q[:, 0])
    vel_list = []
    for i in range(4):
        grad_i = torch.autograd.grad(
            q[:, i], t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True
        )[0]
        vel_list.append(grad_i)
    p = torch.cat(vel_list, dim=1)  # momenta = velocities (m=1)

    # Compute learned Hamiltonian
    H_learned = model.hamiltonian(q, p)  # (batch, 1)

    # Compute true Hamiltonian H = T + V
    x1, z1 = q[:, 0], q[:, 1]
    x2, z2 = q[:, 2], q[:, 3]
    x3 = -(x1 + x2)
    z3 = -(z1 + z2)

    px1, pz1 = p[:, 0], p[:, 1]
    px2, pz2 = p[:, 2], p[:, 3]
    px3 = -(px1 + px2)
    pz3 = -(pz1 + pz2)

    # Kinetic energy T = sum |p_i|^2 / (2*m_i), m=1
    T_true = 0.5 * (px1**2 + pz1**2 + px2**2 + pz2**2 + px3**2 + pz3**2)

    # Potential energy V = -G * sum m_i*m_j / |r_ij|
    eps = 1e-6
    r12 = torch.sqrt((x2 - x1)**2 + (z2 - z1)**2 + eps)
    r13 = torch.sqrt((x3 - x1)**2 + (z3 - z1)**2 + eps)
    r23 = torch.sqrt((x3 - x2)**2 + (z3 - z2)**2 + eps)
    V_true = -G * (1/r12 + 1/r13 + 1/r23)

    H_true = T_true + V_true  # (batch,) — note: H = T + V (not T - V like Lagrangian)

    ham_loss = torch.mean(torch.abs(H_learned.squeeze() - H_true))

    return ham_loss


def train_epoch(model, data_loader, collocation_loader, optimizer,
                criterion, norm_stats, device,
                lambda_physics=1.0, lambda_hamilton=0.1, lambda_ham_sup=1.0):
    """
    Train for one epoch with combined losses.

    Loss: L_total = L_data + lambda_physics * L_physics + lambda_hamilton * L_hamilton + lambda_ham_sup * L_ham_sup

    Where:
    - L_data: Position prediction loss
    - L_physics: ODE residual (Newton's laws)
    - L_hamilton: Hamilton's equations residual
    - L_ham_sup: Hamiltonian supervision (H_learned vs H_true)
    """
    model.train()
    total_loss = 0
    total_data_loss = 0
    total_physics_loss = 0
    total_hamilton_loss = 0
    total_ham_sup_loss = 0
    n_batches = 0

    collocation_iter = iter(collocation_loader)

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Data loss (same as DNN)
        outputs = model(inputs)
        data_loss = criterion(outputs, targets)

        # Get collocation points for physics losses
        try:
            t_coll, ic_coll = next(collocation_iter)
        except StopIteration:
            collocation_iter = iter(collocation_loader)
            t_coll, ic_coll = next(collocation_iter)

        # Physics loss (ODE residual, same as PINN/LNN)
        physics_loss = compute_physics_loss(model, t_coll, ic_coll, norm_stats, device)

        # Hamilton's equations residual (HNN-specific)
        hamilton_loss = compute_hamilton_loss(model, t_coll, ic_coll, norm_stats, device)

        # Hamiltonian supervision loss (H_learned vs H_true = T + V)
        ham_sup_loss = compute_hamiltonian_supervision_loss(model, t_coll, ic_coll, norm_stats, device)

        # Combined loss
        loss = (data_loss +
                lambda_physics * physics_loss +
                lambda_hamilton * hamilton_loss +
                lambda_ham_sup * ham_sup_loss)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_data_loss += data_loss.item()
        total_physics_loss += physics_loss.item()
        total_hamilton_loss += hamilton_loss.item()
        total_ham_sup_loss += ham_sup_loss.item()
        n_batches += 1

    return (total_loss / n_batches, total_data_loss / n_batches,
            total_physics_loss / n_batches, total_hamilton_loss / n_batches,
            total_ham_sup_loss / n_batches)


def validate(model, loader, criterion, device):
    """Validate the model (data loss only, same as DNN/PINN/LNN)."""
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
    num_layers=10,
    activation='tanh',
    hamiltonian_hidden=64,
    hamiltonian_layers=3,
    lambda_physics=0.1,
    lambda_hamilton=0.1,
    lambda_ham_sup=1.0,
    n_collocation=10000,
    scheduler_type='plateau',
    patience=250,
    save_every=500,
    device='cuda',
    use_separable_hamiltonian=True,
    optuna_trial=None
):
    """
    Main HNN training function.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print(f"\nLoading data from {data_path}...")
    train_loader, val_loader, test_loader, collocation_loader, norm_stats = \
        create_hnn_dataloaders(data_path, batch_size=batch_size, n_collocation=n_collocation)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Collocation batches: {len(collocation_loader)}")

    # Model
    model = ThreeBodyHNN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        hamiltonian_hidden=hamiltonian_hidden,
        hamiltonian_layers=hamiltonian_layers,
        use_separable_hamiltonian=use_separable_hamiltonian
    )
    model = model.to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Position net: {num_layers} layers x {hidden_dim} neurons, {activation}")
    print(f"Hamiltonian net: {hamiltonian_layers} layers x {hamiltonian_hidden} neurons")
    print(f"Separable Hamiltonian: {use_separable_hamiltonian}")

    # Loss & Optimizer
    criterion = nn.L1Loss()  # MAE (same as DNN/PINN/LNN)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Config
    config = {
        'data_path': str(data_path),
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'activation': activation,
        'hamiltonian_hidden': hamiltonian_hidden,
        'hamiltonian_layers': hamiltonian_layers,
        'lambda_physics': lambda_physics,
        'lambda_hamilton': lambda_hamilton,
        'lambda_ham_sup': lambda_ham_sup,
        'use_separable_hamiltonian': use_separable_hamiltonian,
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
    print(f"Starting HNN training (lambda_physics={lambda_physics}, lambda_hamilton={lambda_hamilton}, lambda_ham_sup={lambda_ham_sup})...")
    print(f"{'='*70}\n")

    history = {
        'train_loss': [], 'val_loss': [],
        'data_loss': [], 'physics_loss': [], 'hamilton_loss': [], 'ham_sup_loss': [], 'lr': []
    }
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, data_loss, physics_loss, hamilton_loss, ham_sup_loss = train_epoch(
            model, train_loader, collocation_loader, optimizer,
            criterion, norm_stats, device, lambda_physics, lambda_hamilton, lambda_ham_sup
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['data_loss'].append(data_loss)
        history['physics_loss'].append(physics_loss)
        history['hamilton_loss'].append(hamilton_loss)
        history['ham_sup_loss'].append(ham_sup_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Scheduler
        if scheduler_type == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Optuna pruning
        if optuna_trial:
            optuna_trial.report(val_loss, epoch)
            if optuna_trial.should_prune():
                import optuna
                raise optuna.exceptions.TrialPruned()

        # Logging
        if epoch % 100 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:5d}/{epochs} | "
                  f"Loss: {train_loss:.4f} (D:{data_loss:.4f} P:{physics_loss:.4f} H:{hamilton_loss:.4f} Hs:{ham_sup_loss:.4f}) | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")

        # Save best
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
    print("HNN Training Complete!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Final test loss: {test_loss:.6f}")
    print(f"Model saved to: {output_dir}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Three-Body HNN')
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
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['softplus', 'tanh', 'relu', 'elu'])
    parser.add_argument('--hamiltonian-hidden', type=int, default=64)
    parser.add_argument('--hamiltonian-layers', type=int, default=3)
    parser.add_argument('--lambda-physics', type=float, default=0.1,
                        help='Weight for physics (ODE) loss')
    parser.add_argument('--lambda-hamilton', type=float, default=0.1,
                        help='Weight for Hamilton equations loss')
    parser.add_argument('--lambda-ham-sup', type=float, default=1.0,
                        help='Weight for Hamiltonian supervision loss (H_learned vs H_true)')
    parser.add_argument('--no-separable-hamiltonian', action='store_true',
                        help='Disable separable H=T+V structure (use unconstrained Hamiltonian)')
    parser.add_argument('--n-collocation', type=int, default=10000)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'])
    parser.add_argument('--patience', type=int, default=50)
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
        activation=args.activation,
        hamiltonian_hidden=args.hamiltonian_hidden,
        hamiltonian_layers=args.hamiltonian_layers,
        lambda_physics=args.lambda_physics,
        lambda_hamilton=args.lambda_hamilton,
        lambda_ham_sup=args.lambda_ham_sup,
        use_separable_hamiltonian=not args.no_separable_hamiltonian,
        n_collocation=args.n_collocation,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        patience=args.patience,
        device=args.device
    )


if __name__ == '__main__':
    main()
