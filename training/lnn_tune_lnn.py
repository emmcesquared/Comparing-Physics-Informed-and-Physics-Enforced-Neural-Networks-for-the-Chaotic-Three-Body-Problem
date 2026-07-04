import optuna
from optuna.trial import TrialState
import torch
import numpy as np
import argparse
from pathlib import Path
from train import train

def objective(trial):
    # Hyperparameters to tune
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096]),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 2, 8),
        'activation': 'softplus',  # Fixed to softplus for physics
        'lambda_lag': trial.suggest_float('lambda_lag', 0.01, 10.0, log=True),
        'lambda_el': trial.suggest_float('lambda_el', 0.001, 0.5, log=True), # Reduced upper bound slightly
        'epochs': 300, # Significantly reduced for faster tuning
        'patience': 30,
        'scheduler_type': trial.suggest_categorical('scheduler', ['plateau', 'cosine']),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    }

    # Fixed paths
    data_path = '../../dataset generation/pinn_three_body_dataset.npz'
    output_dir = Path(f'experiments/tuning_trial_{trial.number}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTrial {trial.number} Params: {params}")

    try:
        model, history = train(
            data_path=data_path,
            output_dir=output_dir,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            lr=params['lr'],
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            activation=params['activation'],
            lambda_lag=params['lambda_lag'],
            lambda_el=params['lambda_el'],
            lambda_physics=0.1, 
            n_collocation=2000, # Reduced for speed
            scheduler_type=params['scheduler_type'],
            patience=params['patience'],
            weight_decay=params['weight_decay'],
            device='cuda' if torch.cuda.is_available() else 'cpu',
            optuna_trial=trial
        )
        
        val_loss = history['val_loss'][-1]
        
        # Penalize if E-L residual is too high? 
        # For now just optimize validation MSE, assuming physics losses force constraint
        
        return val_loss

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=50)
    args = parser.parse_args()

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
