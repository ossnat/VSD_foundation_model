"""
Full training script for MAE 2D model (single frames) with the new CSV-based dataset structure.

This script:
1. Loads config from configs/default.yaml
2. Creates train/val/test datasets and loaders
3. Builds MAE 2D model (ResNet18 encoder + decoder)
4. Runs full training loop with metrics tracking
5. Evaluates on test set
6. Visualizes predictions (true vs predicted frames)
7. Uses Grad-CAM for model interpretability

Usage:
    python scripts/train_mae_2d.py [--config configs/default.yaml] [--epochs 10]
"""
import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

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
from src.models.backbone.mae_backbone_2d import MAEResNet18Backbone
from src.models.heads.mae_decoder_2d import MAEDecoder2D
from src.models.systems.mae_system import MAESystem
from src.utils.logger import TBLogger, set_seed

# Import helper functions from utils
from src.utils.mae_utils import (
    load_config,
    create_datasets,
    create_dataloaders,
    extract_single_frame,
    create_masked_batch,
    build_mae_2d_model
)


class MAE2DTrainer:
    """Custom trainer for MAE 2D that handles frame extraction and masking."""
    
    def __init__(self, model, optimizer, device, logger=None, use_tpu=False, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.use_tpu = use_tpu
        self.max_grad_norm = max_grad_norm  # Gradient clipping
        # Only use GradScaler for CUDA, not for TPU (TPU uses different precision)
        self.scaler = torch.cuda.amp.GradScaler() if not use_tpu and torch.cuda.is_available() else None
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        # Batch loss values to reduce host-device transfers
        loss_buffer = []
        log_interval = 100  # Log every 100 batches for more frequent updates
        print_interval = 100  # Print detailed info every 100 batches
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Extract single frame and create masked batch
            frame_batch = extract_single_frame(batch)
            mae_batch = create_masked_batch(frame_batch, mask_ratio=0.75)
            
            # Validate input data (check for NaN/Inf)
            for key, tensor in mae_batch.items():
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"\n⚠️ ERROR: NaN/Inf detected in input '{key}' at batch {batch_idx}")
                    print(f"  Tensor stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")
                    print(f"  This suggests a problem with data loading or preprocessing!")
                    raise ValueError(f"NaN/Inf in input data at batch {batch_idx}")
            
            # Move to device
            mae_batch = {k: v.to(self.device, non_blocking=True) for k, v in mae_batch.items()}
            
            # Forward pass with mixed precision
            if self.use_tpu:
                # TPU uses bfloat16 by default, no need for autocast
                output = self.model(mae_batch)
                loss = output["loss"]
                
                # Check for NaN loss immediately
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n❌ NaN/Inf loss detected at batch {batch_idx}!")
                    print(f"  Loss value: {loss.item()}")
                    print(f"  Input stats: video_masked min={mae_batch['video_masked'].min().item():.4f}, max={mae_batch['video_masked'].max().item():.4f}")
                    print(f"  Model parameters check:")
                    for name, param in self.model.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            print(f"    {name}: has NaN/Inf")
                    raise ValueError(f"NaN loss detected at batch {batch_idx}. Stopping training.")
                
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    import torch_xla.core.xla_model as xm
                    xm.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # For TPU, use xm.optimizer_step
                import torch_xla.core.xla_model as xm
                xm.optimizer_step(self.optimizer)
            else:
                # CUDA mixed precision
                with torch.cuda.amp.autocast():
                    output = self.model(mae_batch)
                    loss = output["loss"]
                
                # Check for NaN loss immediately
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n❌ NaN/Inf loss detected at batch {batch_idx}!")
                    print(f"  Loss value: {loss.item()}")
                    print(f"  Input stats: video_masked min={mae_batch['video_masked'].min().item():.4f}, max={mae_batch['video_masked'].max().item():.4f}")
                    print(f"  Model parameters check:")
                    for name, param in self.model.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            print(f"    {name}: has NaN/Inf")
                    raise ValueError(f"NaN loss detected at batch {batch_idx}. Stopping training.")
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping (unscale first)
                    self.scaler.unscale_(self.optimizer)
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
            
            # Store loss in buffer (avoid immediate .item() call)
            loss_buffer.append(loss.detach())
            num_batches += 1
            
            # Update progress bar and log less frequently to reduce host-device transfers
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                # Compute average from buffer
                avg_batch_loss = torch.stack(loss_buffer).mean()
                loss_val = avg_batch_loss.item()
                
                # Check for NaN in accumulated loss
                if np.isnan(loss_val) or np.isinf(loss_val):
                    print(f"\n❌ NaN/Inf detected in accumulated loss at batch {batch_idx+1}!")
                    print(f"  Average loss: {loss_val}")
                    print(f"  Buffer size: {len(loss_buffer)}")
                    raise ValueError(f"NaN loss detected. Stopping training.")
                
                total_loss += loss_val * len(loss_buffer)
                loss_buffer.clear()
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})
                
                # Detailed print every 100 batches
                if (batch_idx + 1) % print_interval == 0:
                    print(f"\n📊 Batch {batch_idx+1}/{len(train_loader)} (Epoch {epoch+1})")
                    print(f"  Loss: {loss_val:.6f}")
                    if "metrics" in output:
                        for key, value in output["metrics"].items():
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:.6f}")
                    # Check gradient norms
                    total_norm = 0.0
                    param_count = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                    total_norm = total_norm ** (1. / 2)
                    print(f"  Gradient norm: {total_norm:.6f}")
                    print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Log to tensorboard (less frequently)
                if self.logger:
                    global_step = epoch * len(train_loader) + batch_idx + 1
                    self.logger.log_scalar("train/loss", loss_val, global_step)
                    if "metrics" in output:
                        for key, value in output["metrics"].items():
                            # Only log scalar metrics (avoid .item() for tensors)
                            if isinstance(value, (int, float)):
                                self.logger.log_scalar(f"train/{key}", value, global_step)
        
        # Handle any remaining losses in buffer
        if loss_buffer:
            avg_batch_loss = torch.stack(loss_buffer).mean()
            total_loss += avg_batch_loss.item() * len(loss_buffer)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, val_loader, epoch):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_metrics = {}
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            loss_buffer = []
            for batch_idx, batch in enumerate(pbar):
                # Extract single frame and create masked batch
                frame_batch = extract_single_frame(batch)
                mae_batch = create_masked_batch(frame_batch, mask_ratio=0.75)
                
                # Move to device
                mae_batch = {k: v.to(self.device, non_blocking=True) for k, v in mae_batch.items()}
                
                # Forward pass
                if self.use_tpu:
                    output = self.model(mae_batch)
                    loss = output["loss"]
                else:
                    with torch.cuda.amp.autocast():
                        output = self.model(mae_batch)
                        loss = output["loss"]
                
                # Store loss in buffer (avoid immediate .item() call)
                loss_buffer.append(loss.detach())
                num_batches += 1
                
                # Update progress bar less frequently
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                    avg_batch_loss = torch.stack(loss_buffer).mean()
                    loss_val = avg_batch_loss.item()
                    total_loss += loss_val * len(loss_buffer)
                    loss_buffer.clear()
                    pbar.set_postfix({"loss": f"{loss_val:.4f}"})
                
                # Accumulate metrics (only scalar values)
                if "metrics" in output:
                    for key, value in output["metrics"].items():
                        if isinstance(value, (int, float)):
                            if key not in all_metrics:
                                all_metrics[key] = []
                            all_metrics[key].append(value)
        
        # Handle any remaining losses in buffer
        if loss_buffer:
            avg_batch_loss = torch.stack(loss_buffer).mean()
            total_loss += avg_batch_loss.item() * len(loss_buffer)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        
        # Log to tensorboard
        if self.logger:
            self.logger.log_scalar("val/loss", avg_loss, epoch)
            for key, value in avg_metrics.items():
                self.logger.log_scalar(f"val/{key}", value, epoch)
        
        return avg_loss, avg_metrics


def evaluate_test_set(model, test_loader, device):
    """Evaluate model on test set and return detailed metrics."""
    model.eval()
    test_losses = []
    all_metrics = {}
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            # Extract single frame and create masked batch
            frame_batch = extract_single_frame(batch)
            mae_batch = create_masked_batch(frame_batch, mask_ratio=0.75)
            
            # Move to device
            mae_batch = {k: v.to(device) for k, v in mae_batch.items()}
            
            # Forward pass
            output = model(mae_batch)
            loss = output.get("loss", None)
            
            if loss is not None:
                test_losses.append(float(loss.item()))
            
            # Accumulate metrics
            if "metrics" in output:
                for key, value in output["metrics"].items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
    
    # Calculate averages
    avg_loss = np.mean(test_losses) if test_losses else 0.0
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    print(f"\nTest Results:")
    print(f"  Average Loss: {avg_loss:.4f}")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return avg_loss, avg_metrics


def visualize_predictions(model, test_loader, device, num_examples=5, save_dir="visualizations"):
    """Visualize model predictions on test examples."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nVisualizing {num_examples} predictions...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_examples:
                break
            
            # Extract single frame and create masked batch
            frame_batch = extract_single_frame(batch)
            mae_batch = create_masked_batch(frame_batch, mask_ratio=0.75)
            
            # Move to device
            mae_batch = {k: v.to(device) for k, v in mae_batch.items()}
            
            # Get predictions
            output = model(mae_batch)
            
            # Extract tensors
            video_target = mae_batch["video_target"].cpu()  # (B, C, H, W)
            video_masked = mae_batch["video_masked"].cpu()
            mask = mae_batch["mask"].cpu()
            
            # Get reconstruction by doing forward pass
            with torch.no_grad():
                features = model.encoder(video_masked.to(device))
                reconstruction = model.decoder(features, target_size=(video_target.shape[2], video_target.shape[3]))
                reconstruction = reconstruction.cpu()
            
            # Take first item in batch
            target = video_target[0, 0].numpy()  # (H, W)
            masked = video_masked[0, 0].numpy()
            recon = reconstruction[0, 0].numpy()
            mask_np = mask[0, 0].numpy()
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            # Original (target)
            im1 = axes[0, 0].imshow(target, cmap='hot', vmin=target.min(), vmax=target.max())
            axes[0, 0].set_title("Original Frame")
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Masked input
            im2 = axes[0, 1].imshow(masked, cmap='hot', vmin=masked.min(), vmax=masked.max())
            axes[0, 1].set_title("Masked Input")
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Reconstruction
            im3 = axes[1, 0].imshow(recon, cmap='hot', vmin=recon.min(), vmax=recon.max())
            axes[1, 0].set_title("Reconstruction")
            axes[1, 0].axis('off')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # Error (difference)
            error = np.abs(target - recon)
            im4 = axes[1, 1].imshow(error, cmap='hot', vmin=error.min(), vmax=error.max())
            axes[1, 1].set_title("Absolute Error")
            axes[1, 1].axis('off')
            plt.colorbar(im4, ax=axes[1, 1])
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"prediction_example_{i+1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved visualization {i+1} to {save_path}")


class GradCAM:
    """Grad-CAM implementation for MAE 2D model."""
    
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer = None
        
        # Find the last convolutional layer in encoder
        self._find_target_layer()
        self._hook_layers()
    
    def _find_target_layer(self):
        """Find the last Conv2d layer in the encoder."""
        last_conv = None
        last_conv_name = None
        
        for name, module in self.model.encoder.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
                last_conv_name = name
        
        if last_conv is None:
            raise ValueError("Could not find Conv2d layer in encoder")
        
        self.target_layer = last_conv
        print(f"  Using layer for Grad-CAM: {last_conv_name}")
    
    def _hook_layers(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_tensor):
        """Generate Grad-CAM heatmap."""
        self.model.train()  # Need gradients
        
        # Clear previous gradients
        self.gradients = None
        self.activations = None
        
        # Forward pass
        features = self.model.encoder(input_tensor)
        reconstruction = self.model.decoder(features, target_size=(target_tensor.shape[2], target_tensor.shape[3]))
        
        # Compute loss
        loss = F.mse_loss(reconstruction, target_tensor)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Check if we got gradients and activations
        if self.gradients is None or self.activations is None:
            raise ValueError("Failed to capture gradients or activations")
        
        # Get gradients and activations (take first item in batch)
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # (C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0, keepdim=True)  # (1, H, W)
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        # Upsample to input size
        cam = F.interpolate(cam.unsqueeze(0), size=(target_tensor.shape[2], target_tensor.shape[3]), 
                           mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()  # (H, W)


def visualize_gradcam(model, test_loader, device, num_examples=3, save_dir="visualizations"):
    """Visualize Grad-CAM heatmaps for model interpretability."""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nGenerating Grad-CAM visualizations for {num_examples} examples...")
    
    try:
        gradcam = GradCAM(model)
    except Exception as e:
        print(f"  Warning: Could not initialize Grad-CAM: {e}. Skipping.")
        return
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_examples:
                break
            
            # Extract single frame
            frame_batch = extract_single_frame(batch)
            video = frame_batch["video"].to(device)  # (B, C, H, W)
            
            # Create masked version
            mae_batch = create_masked_batch(frame_batch, mask_ratio=0.75)
            video_masked = mae_batch["video_masked"].to(device)
            video_target = mae_batch["video_target"].to(device)
            
            # Generate CAM
            try:
                cam = gradcam.generate_cam(video_masked, video_target)
            except Exception as e:
                print(f"  Warning: Could not generate CAM for example {i+1}: {e}")
                continue
            
            # Get reconstruction (set model back to eval for inference)
            model.eval()
            with torch.no_grad():
                features = model.encoder(video_masked)
                reconstruction = model.decoder(features, target_size=(video_target.shape[2], video_target.shape[3]))
                reconstruction = reconstruction.cpu()[0, 0].numpy()
            
            target = video_target.cpu()[0, 0].numpy()
            masked = video_masked.cpu()[0, 0].numpy()
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            # Original
            im1 = axes[0, 0].imshow(target, cmap='hot')
            axes[0, 0].set_title("Original Frame")
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Grad-CAM heatmap
            im2 = axes[0, 1].imshow(cam, cmap='jet', alpha=0.7)
            axes[0, 1].set_title("Grad-CAM Heatmap")
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Overlay
            axes[1, 0].imshow(target, cmap='gray', alpha=0.5)
            axes[1, 0].imshow(cam, cmap='jet', alpha=0.5)
            axes[1, 0].set_title("Grad-CAM Overlay")
            axes[1, 0].axis('off')
            
            # Reconstruction
            im4 = axes[1, 1].imshow(reconstruction, cmap='hot')
            axes[1, 1].set_title("Reconstruction")
            axes[1, 1].axis('off')
            plt.colorbar(im4, ax=axes[1, 1])
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"gradcam_example_{i+1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved Grad-CAM visualization {i+1} to {save_path}")


def save_checkpoint(model, optimizer, epoch, loss, save_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"  Saved best model to {best_path}")


def train_mae_2d_from_config(config_path=None, cfg=None, epochs=None, 
                             save_dir="checkpoints", log_dir="logs", vis_dir="visualizations"):
    """
    Train MAE 2D model from configuration.
    
    This function can be called directly from Colab or other environments.
    
    Args:
        config_path: Path to config YAML file (if cfg is None)
        cfg: Configuration dictionary (if provided, config_path is ignored)
        epochs: Number of epochs (overrides config if provided)
        save_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        vis_dir: Directory to save visualizations
    
    Returns:
        Trained model
    """
    print("="*60)
    print("MAE 2D Full Training Script")
    print("="*60)
    
    # Load config if not provided
    if cfg is None:
        if config_path is None:
            config_path = project_root / "configs" / "default.yaml"
        else:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                config_path = project_root / config_path
        cfg = load_config(config_path)
    
    # Override epochs if provided
    if epochs is not None:
        cfg["epochs"] = epochs
    
    # Set seed and device
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Check for TPU support (for Colab TPU)
    use_tpu = False
    if hasattr(torch, 'xla') and torch.xla.is_available():
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        use_tpu = True
        print(f"TPU detected! Using XLA device: {device}")
        print(f"TPU has {torch.xla.device_count()} cores")
    
    # Create datasets (use full datasets, not limited)
    print(f"\nCreating datasets...")
    train_dataset_full = load_dataset(cfg, split="train", batch_size=1, num_workers=4, shuffle=False).dataset
    val_dataset_full = load_dataset(cfg, split="val", batch_size=1, num_workers=4, shuffle=False).dataset
    test_dataset_full = load_dataset(cfg, split="test", batch_size=1, num_workers=4, shuffle=False).dataset
    
    print(f"  Dataset sizes: train={len(train_dataset_full)}, val={len(val_dataset_full)}, test={len(test_dataset_full)}")
    
    # Create DataLoaders with optimizations for Colab/TPU
    batch_size = cfg.get("batch_size", 256)  # Default to 256 for TPU efficiency (multiple of 8)
    num_workers = cfg.get("num_workers", 4)
    pin_memory = cfg.get("pin_memory", True)  # Faster GPU transfer
    persistent_workers = cfg.get("persistent_workers", True)  # Keep workers alive between epochs
    prefetch_factor = cfg.get("prefetch_factor", 2)  # Prefetch batches
    
    train_loader = DataLoader(
        train_dataset_full, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset_full, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    test_loader = DataLoader(
        test_dataset_full, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    # Build model
    model = build_mae_2d_model(cfg, device)
    
    # Create optimizer
    optimizer = model.get_optimizer(
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.05)
    )
    
    # Create logger
    logger = TBLogger(log_dir=log_dir)
    
    # Create trainer with gradient clipping
    max_grad_norm = cfg.get("max_grad_norm", 1.0)  # Gradient clipping to prevent explosion
    trainer = MAE2DTrainer(model, optimizer, device, logger, use_tpu=use_tpu, max_grad_norm=max_grad_norm)
    print(f"  Gradient clipping: max_norm={max_grad_norm}")
    
    # Training loop
    num_epochs = cfg.get("epochs", 10)
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_loss, val_metrics = trainer.validate(val_loader, epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        if val_metrics:
            for key, value in val_metrics.items():
                print(f"  Val {key}: {value:.4f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint(model, optimizer, epoch, val_loss, save_dir, is_best=is_best)
        
        print("-"*60)
    
    # Evaluate on test set
    print("\n" + "="*60)
    test_loss, test_metrics = evaluate_test_set(model, test_loader, device)
    
    # Visualize predictions
    print("\n" + "="*60)
    visualize_predictions(model, test_loader, device, num_examples=5, save_dir=vis_dir)
    
    # Grad-CAM visualization
    print("\n" + "="*60)
    visualize_gradcam(model, test_loader, device, num_examples=3, save_dir=vis_dir)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"  Checkpoints saved to: {save_dir}")
    print(f"  Visualizations saved to: {vis_dir}")
    print(f"  TensorBoard logs: {log_dir}")
    print("="*60)
    
    return model


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Train MAE 2D model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory for TensorBoard logs")
    parser.add_argument("--vis-dir", type=str, default="visualizations",
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Call the main training function
    train_mae_2d_from_config(
        config_path=args.config,
        epochs=args.epochs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        vis_dir=args.vis_dir
    )


if __name__ == "__main__":
    main()
# Run from command line: 
# python scripts/train_mae_2d.py --epochs 10
# Run in Colab: 
# train_mae_2d_from_config(cfg_dict, epochs=10)

