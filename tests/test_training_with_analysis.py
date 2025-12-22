"""
Training test with gradient analysis and visualization.

This test:
1. Loads config from configs/default.yaml
2. Creates train/val/test datasets (1000/500/500 samples)
3. Analyzes gradients before full training (exploding/vanishing detection)
4. Trains model with batch size 128
5. Prints loss per batch for train and val
6. Evaluates on test set with metrics
7. Visualizes gradient behavior and loss curves

Usage:
    python tests/test_training_with_analysis.py
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
# Handle both script execution and Colab notebook execution
try:
    # When running as a script
    project_root = Path(__file__).parent.parent
except NameError:
    # When running in Colab/notebook, __file__ is not defined
    import os
    cwd = Path(os.getcwd())
    if (cwd / "configs").exists():
        project_root = cwd
    elif (cwd.parent / "configs").exists():
        project_root = cwd.parent
    else:
        project_root = cwd
        print(f"Warning: Could not determine project root, using: {project_root}")

sys.path.insert(0, str(project_root))

from src.data import load_dataset
from src.models import build_ssl_model
from src.utils.mae_utils import extract_single_frame, create_masked_batch

# Try to import logger, make it optional
try:
    from src.utils.logger import set_seed
    HAS_LOGGER = True
except ImportError:
    print("Warning: Logger not available, using basic seed setting")
    HAS_LOGGER = False
    def set_seed(seed):
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def create_limited_dataset(dataset, max_samples):
    """Create a subset of the dataset with at most max_samples."""
    actual_samples = min(len(dataset), max_samples)
    indices = list(range(actual_samples))
    return Subset(dataset, indices)


def analyze_gradients(model, train_loader, device, max_batches=10):
    """
    Analyze gradients for exploding/vanishing gradient problems.
    
    Returns:
        dict with gradient statistics
    """
    print("\n" + "="*60)
    print("Gradient Analysis (1 epoch)")
    print("="*60)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Track gradients
    grad_stats = defaultdict(list)
    param_names = []
    
    # Collect parameter names once
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_names.append(name)
    
    batch_losses = []
    
    print(f"\nRunning {max_batches} batches for gradient analysis...")
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Gradient analysis", total=max_batches)):
        if batch_idx >= max_batches:
            break
        
        optimizer.zero_grad()
        
        # Extract single frame and create masked batch
        frame_batch = extract_single_frame(batch)
        mae_batch = create_masked_batch(frame_batch, mask_ratio=0.75)
        
        # Move to device
        mae_batch = {k: v.to(device, non_blocking=True) for k, v in mae_batch.items()}
        
        # Forward pass
        output = model(mae_batch)
        loss = output["loss"]
        
        # Backward pass
        loss.backward()
        
        # Collect gradient statistics
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.data
                grad_norm = grad.norm().item()
                grad_max = grad.abs().max().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                
                grad_stats[name].append({
                    'norm': grad_norm,
                    'max': grad_max,
                    'mean': grad_mean,
                    'std': grad_std
                })
        
        batch_losses.append(loss.item())
        
        optimizer.step()
    
    # Analyze gradient statistics
    print(f"\n{'='*60}")
    print("Gradient Statistics Summary")
    print(f"{'='*60}")
    
    summary_stats = {}
    for name in param_names:
        if name in grad_stats and len(grad_stats[name]) > 0:
            norms = [s['norm'] for s in grad_stats[name]]
            maxs = [s['max'] for s in grad_stats[name]]
            
            avg_norm = np.mean(norms)
            max_norm = np.max(norms)
            avg_max = np.mean(maxs)
            
            summary_stats[name] = {
                'avg_norm': avg_norm,
                'max_norm': max_norm,
                'avg_max': avg_max
            }
            
            # Check for problems
            status = "✓"
            if max_norm > 100:
                status = "⚠️ EXPLODING"
            elif max_norm < 1e-6:
                status = "⚠️ VANISHING"
            
            print(f"{status} {name[:50]:<50} | avg_norm: {avg_norm:.4e} | max_norm: {max_norm:.4e}")
    
    # Overall gradient health
    all_max_norms = [s['max_norm'] for s in summary_stats.values()]
    overall_max = max(all_max_norms) if all_max_norms else 0
    overall_avg = np.mean(all_max_norms) if all_max_norms else 0
    
    print(f"\n{'='*60}")
    print("Overall Gradient Health:")
    print(f"  Maximum gradient norm across all layers: {overall_max:.4e}")
    print(f"  Average gradient norm across all layers: {overall_avg:.4e}")
    
    if overall_max > 100:
        print(f"  ⚠️ WARNING: Exploding gradients detected! (max > 100)")
    elif overall_max < 1e-6:
        print(f"  ⚠️ WARNING: Vanishing gradients detected! (max < 1e-6)")
    else:
        print(f"  ✓ Gradients look healthy")
    
    return {
        'grad_stats': grad_stats,
        'summary_stats': summary_stats,
        'batch_losses': batch_losses,
        'overall_max_norm': overall_max,
        'overall_avg_norm': overall_avg
    }


def plot_gradient_analysis(grad_analysis, save_dir="test_analysis_outputs"):
    """Plot gradient behavior over batches."""
    os.makedirs(save_dir, exist_ok=True)
    
    grad_stats = grad_analysis['grad_stats']
    
    # Plot 1: Gradient norms over batches for key layers
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Select a few representative layers to plot
    layer_names = list(grad_stats.keys())
    # Prioritize: encoder, decoder, first/last layers
    selected_layers = []
    for pattern in ['encoder', 'decoder', 'input_conv', 'weight', 'bias']:
        for name in layer_names:
            if pattern in name.lower() and name not in selected_layers:
                selected_layers.append(name)
                if len(selected_layers) >= 4:
                    break
        if len(selected_layers) >= 4:
            break
    
    # Fill remaining slots
    while len(selected_layers) < 4 and len(selected_layers) < len(layer_names):
        for name in layer_names:
            if name not in selected_layers:
                selected_layers.append(name)
                break
    
    for idx, layer_name in enumerate(selected_layers[:4]):
        ax = axes[idx // 2, idx % 2]
        if layer_name in grad_stats:
            norms = [s['norm'] for s in grad_stats[layer_name]]
            maxs = [s['max'] for s in grad_stats[layer_name]]
            
            ax.plot(norms, label='Gradient Norm', linewidth=2)
            ax.plot(maxs, label='Max Gradient', linewidth=2, linestyle='--')
            ax.set_title(f"{layer_name[:40]}...", fontsize=10)
            ax.set_xlabel("Batch")
            ax.set_ylabel("Gradient Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "gradient_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved gradient analysis plot to {save_path}")
    
    # Plot 2: Gradient norm distribution across all layers
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    summary_stats = grad_analysis['summary_stats']
    layer_names_short = [name.split('.')[-1] for name in summary_stats.keys()]
    avg_norms = [s['avg_norm'] for s in summary_stats.values()]
    
    # Sort by norm for better visualization
    sorted_data = sorted(zip(layer_names_short, avg_norms), key=lambda x: x[1], reverse=True)
    layer_names_short, avg_norms = zip(*sorted_data) if sorted_data else ([], [])
    
    # Show top 20 layers
    top_n = min(20, len(layer_names_short))
    ax.barh(range(top_n), avg_norms[:top_n])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([name[:30] for name in layer_names_short[:top_n]], fontsize=8)
    ax.set_xlabel("Average Gradient Norm")
    ax.set_title("Gradient Norms by Layer (Top 20)")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "gradient_norms_by_layer.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved gradient norms plot to {save_path}")


def train_with_monitoring(model, train_loader, val_loader, device, epochs=3, 
                          optimizer=None, max_grad_norm=1.0):
    """Train model with per-batch loss printing."""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        epoch_train_losses = []
        
        print(f"\nTraining (batch size {train_loader.batch_size})...")
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # Extract single frame and create masked batch
            frame_batch = extract_single_frame(batch)
            mae_batch = create_masked_batch(frame_batch, mask_ratio=0.75)
            
            # Move to device
            mae_batch = {k: v.to(device, non_blocking=True) for k, v in mae_batch.items()}
            
            # Forward pass
            output = model(mae_batch)
            loss = output["loss"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            loss_val = loss.item()
            epoch_train_losses.append(loss_val)
            train_losses.append(loss_val)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            
            # Print every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss = {loss_val:.6f}")
        
        avg_train_loss = np.mean(epoch_train_losses)
        print(f"\n  Average Train Loss: {avg_train_loss:.6f}")
        
        # Validation
        model.eval()
        epoch_val_losses = []
        
        print(f"\nValidation (batch size {val_loader.batch_size})...")
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                # Extract single frame and create masked batch
                frame_batch = extract_single_frame(batch)
                mae_batch = create_masked_batch(frame_batch, mask_ratio=0.75)
                
                # Move to device
                mae_batch = {k: v.to(device, non_blocking=True) for k, v in mae_batch.items()}
                
                # Forward pass
                output = model(mae_batch)
                loss = output["loss"]
                
                loss_val = loss.item()
                epoch_val_losses.append(loss_val)
                val_losses.append(loss_val)
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})
                
                # Print every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx+1}/{len(val_loader)}: Loss = {loss_val:.6f}")
        
        avg_val_loss = np.mean(epoch_val_losses)
        print(f"\n  Average Val Loss: {avg_val_loss:.6f}")
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
    
    return train_losses, val_losses


def evaluate_test_set(model, test_loader, device):
    """Evaluate model on test set with multiple metrics."""
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")
    
    model.eval()
    
    all_losses = []
    all_mse_overall = []
    all_mse_masked = []
    all_mse_visible = []
    
    print(f"\nEvaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test evaluation"):
            # Extract single frame and create masked batch
            frame_batch = extract_single_frame(batch)
            mae_batch = create_masked_batch(frame_batch, mask_ratio=0.75)
            
            # Move to device
            mae_batch = {k: v.to(device, non_blocking=True) for k, v in mae_batch.items()}
            
            # Forward pass
            output = model(mae_batch)
            
            loss = output["loss"].item()
            all_losses.append(loss)
            
            if "metrics" in output:
                metrics = output["metrics"]
                all_mse_overall.append(metrics.get("mse_overall", 0))
                all_mse_masked.append(metrics.get("mse_masked", 0))
                all_mse_visible.append(metrics.get("mse_visible", 0))
    
    # Compute statistics
    results = {
        'loss': {
            'mean': np.mean(all_losses),
            'std': np.std(all_losses),
            'min': np.min(all_losses),
            'max': np.max(all_losses)
        }
    }
    
    if all_mse_overall:
        results['mse_overall'] = {
            'mean': np.mean(all_mse_overall),
            'std': np.std(all_mse_overall)
        }
        results['mse_masked'] = {
            'mean': np.mean(all_mse_masked),
            'std': np.std(all_mse_masked)
        }
        results['mse_visible'] = {
            'mean': np.mean(all_mse_visible),
            'std': np.std(all_mse_visible)
        }
    
    # Print results
    print(f"\nTest Set Metrics:")
    print(f"  Loss: {results['loss']['mean']:.6f} ± {results['loss']['std']:.6f}")
    print(f"    Range: [{results['loss']['min']:.6f}, {results['loss']['max']:.6f}]")
    
    if 'mse_overall' in results:
        print(f"  MSE Overall: {results['mse_overall']['mean']:.6f} ± {results['mse_overall']['std']:.6f}")
        print(f"  MSE Masked: {results['mse_masked']['mean']:.6f} ± {results['mse_masked']['std']:.6f}")
        print(f"  MSE Visible: {results['mse_visible']['mean']:.6f} ± {results['mse_visible']['std']:.6f}")
    
    return results


def plot_loss_curves(train_losses, val_losses, save_dir="test_analysis_outputs"):
    """Plot training and validation loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Loss per batch
    ax1 = axes[0]
    ax1.plot(train_losses, label='Train Loss', alpha=0.7, linewidth=1)
    ax1.plot(val_losses, label='Val Loss', alpha=0.7, linewidth=1)
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss per Batch (Training and Validation)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Smoothed loss (moving average)
    ax2 = axes[1]
    window = 50
    if len(train_losses) > window:
        train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        val_smooth = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        ax2.plot(train_smooth, label=f'Train Loss (MA{window})', linewidth=2)
        ax2.plot(val_smooth, label=f'Val Loss (MA{window})', linewidth=2)
    else:
        ax2.plot(train_losses, label='Train Loss', linewidth=2)
        ax2.plot(val_losses, label='Val Loss', linewidth=2)
    
    ax2.set_xlabel("Batch (smoothed)")
    ax2.set_ylabel("Loss")
    ax2.set_title("Smoothed Loss Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved loss curves plot to {save_path}")


def test_training_with_analysis():
    """Main test function."""
    print("="*60)
    print("Training Test with Gradient Analysis")
    print("="*60)
    
    # Load config
    config_path = project_root / "configs" / "default.yaml"
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return False
    
    print(f"\n[1] Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Check if using new CSV structure
    if 'split_csv_path' not in cfg or 'stats_json_path' not in cfg:
        print("ERROR: Config must contain 'split_csv_path' and 'stats_json_path'")
        return False
    
    # Verify paths exist
    split_csv_path = Path(cfg['split_csv_path'])
    stats_json_path = Path(cfg['stats_json_path'])
    
    if not split_csv_path.exists():
        print(f"\nERROR: Split CSV not found: {split_csv_path}")
        return False
    
    if not stats_json_path.exists():
        print(f"\nERROR: Stats JSON not found: {stats_json_path}")
        return False
    
    print(f"  ✓ Config loaded successfully")
    
    # Set seed
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2] Using device: {device}")
    
    # Create datasets with specified sample sizes
    print(f"\n[3] Creating datasets...")
    print(f"  Train: 1000 samples, Val: 500 samples, Test: 500 samples")
    
    # Create full datasets first
    train_dataset_full = load_dataset(cfg, split="train", batch_size=1, num_workers=0, shuffle=False).dataset
    val_dataset_full = load_dataset(cfg, split="val", batch_size=1, num_workers=0, shuffle=False).dataset
    test_dataset_full = load_dataset(cfg, split="test", batch_size=1, num_workers=0, shuffle=False).dataset
    
    print(f"  Full dataset sizes: train={len(train_dataset_full)}, val={len(val_dataset_full)}, test={len(test_dataset_full)}")
    
    # Create limited subsets
    train_dataset = create_limited_dataset(train_dataset_full, max_samples=1000)
    val_dataset = create_limited_dataset(val_dataset_full, max_samples=500)
    test_dataset = create_limited_dataset(test_dataset_full, max_samples=500)
    
    print(f"  Limited dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Create DataLoaders with batch size 128
    batch_size = 128
    print(f"\n[4] Creating DataLoaders with batch_size={batch_size}...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=cfg.get("pin_memory", True),
        persistent_workers=cfg.get("persistent_workers", True) if cfg.get("num_workers", 0) > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=cfg.get("pin_memory", True),
        persistent_workers=cfg.get("persistent_workers", True) if cfg.get("num_workers", 0) > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=cfg.get("pin_memory", True),
        persistent_workers=cfg.get("persistent_workers", True) if cfg.get("num_workers", 0) > 0 else False
    )
    
    print(f"  ✓ DataLoaders created")
    
    # Build model
    print(f"\n[5] Building model...")
    try:
        model = build_ssl_model(cfg).to(device)
        print(f"  ✓ Model created successfully")
        print(f"    Model type: {type(model).__name__}")
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Gradient analysis (1 epoch)
    print(f"\n[6] Running gradient analysis...")
    try:
        grad_analysis = analyze_gradients(model, train_loader, device, max_batches=10)
        
        # Plot gradient analysis
        print(f"\n[7] Plotting gradient analysis...")
        plot_gradient_analysis(grad_analysis)
        
    except Exception as e:
        print(f"  ✗ Gradient analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Training with monitoring
    print(f"\n[8] Starting training with monitoring...")
    try:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.05)
        )
        
        epochs = cfg.get("epochs", 3)
        max_grad_norm = cfg.get("max_grad_norm", 1.0)
        
        train_losses, val_losses = train_with_monitoring(
            model, train_loader, val_loader, device,
            epochs=epochs, optimizer=optimizer, max_grad_norm=max_grad_norm
        )
        
        # Plot loss curves
        print(f"\n[9] Plotting loss curves...")
        plot_loss_curves(train_losses, val_losses)
        
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Evaluate on test set
    print(f"\n[10] Evaluating on test set...")
    try:
        test_results = evaluate_test_set(model, test_loader, device)
    except Exception as e:
        print(f"  ✗ Test evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"  Gradient Analysis:")
    print(f"    Max gradient norm: {grad_analysis['overall_max_norm']:.4e}")
    print(f"    Avg gradient norm: {grad_analysis['overall_avg_norm']:.4e}")
    print(f"  Training:")
    print(f"    Final train loss: {train_losses[-1]:.6f}")
    print(f"    Final val loss: {val_losses[-1]:.6f}")
    print(f"  Test Set:")
    print(f"    Test loss: {test_results['loss']['mean']:.6f} ± {test_results['loss']['std']:.6f}")
    if 'mse_overall' in test_results:
        print(f"    MSE overall: {test_results['mse_overall']['mean']:.6f}")
        print(f"    MSE masked: {test_results['mse_masked']['mean']:.6f}")
    print(f"\n  All plots saved to: test_analysis_outputs/")
    print(f"{'='*60}")
    
    return True


if __name__ == "__main__":
    success = test_training_with_analysis()
    sys.exit(0 if success else 1)

