# ==================================
# File: src/utils/visualization.py
# ==================================

import torch
try:
    import torchvision.utils as vutils
except Exception:  # make optional for environments without torchvision
    vutils = None
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union


def save_reconstruction_grid(target, recon, mask, fname="recon.png"):
    """Save grid image with [target | recon] for the first item in batch (first 8 frames).
    target, recon: (B,C,T,H,W) tensors in [-inf, inf]
    mask: (B, N) mask in {0,1} – used only for filename annotation here.
    """
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with torch.no_grad():
        tgt = target[0, :, :8]  # (C, T, H, W)
        rec = recon[0, :, :8]
        # make image grids per time, then concatenate along width
        rows = []
        for t in range(tgt.shape[1]):
            row = torch.cat([tgt[:, t], rec[:, t]], dim=-1)  # (C, H, 2W)
            rows.append(row)
        grid = torch.cat(rows, dim=-2)  # (C, 8*H, 2W)
        if vutils is None:
            raise ImportError("torchvision is required for save_reconstruction_grid. Please install torchvision.")
        vutils.save_image((grid - grid.min())/(grid.max()-grid.min()+1e-6), fname)


def plot_frames_sequence(
    video_tensor: torch.Tensor,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    clipping: Optional[Tuple[float, float]] = None,
    cols: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Video Frames Sequence",
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    real_frame_range: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot a sequence of video frames in a grid layout.
    
    This function replicates steps 4-6 from the original visualization code:
    4. Define the frame range to extract for plotting
    5. Extract the desired frame range
    6. Plot the frames as a matrix
    
    Args:
        video_tensor: Video tensor of shape (C, T, H, W) or (T, H, W)
        start_frame: Starting frame index within the tensor (default: 0)
        end_frame: Ending frame index within the tensor (if None, uses all frames from start_frame)
        clipping: Tuple of (min, max) values for color scaling (default: (-0.003, 0.003))
        cols: Number of columns in the grid (default: 10)
        figsize: Figure size as (width, height) (default: auto-calculated)
        title: Title for the plot (default: "Video Frames Sequence")
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot (default: True)
        real_frame_range: Tuple of (real_start, real_end) representing absolute frame indices from original dataset
        
    Returns:
        matplotlib Figure object
    """
    # Step 4: Define the frame range to extract for plotting
    # start_frame and end_frame are relative to the video_tensor
    if end_frame is None:
        if video_tensor.ndim == 4:  # (C, T, H, W)
            end_frame = video_tensor.shape[1] - 1
        else:  # (T, H, W)
            end_frame = video_tensor.shape[0] - 1
    
    # Ensure the frame range is within the bounds of the video tensor
    if start_frame < 0:
        print(f"Warning: start_frame {start_frame} is negative. Setting to 0.")
        start_frame = 0
    
    max_frames = video_tensor.shape[1] if video_tensor.ndim == 4 else video_tensor.shape[0]
    if end_frame >= max_frames:
        print(f"Warning: end_frame {end_frame} is out of bounds for video tensor with {max_frames} frames. Setting to {max_frames - 1}.")
        end_frame = max_frames - 1
    
    if start_frame > end_frame:
        print(f"Warning: start_frame {start_frame} > end_frame {end_frame}. Plotting all frames.")
        start_frame = 0
        end_frame = max_frames - 1
    
    # Step 5: Extract the desired frame range from video_tensor
    if video_tensor.ndim == 4:  # (C, T, H, W)
        frames_to_plot = video_tensor[:, start_frame:end_frame + 1, :, :]
        # Remove the channel dimension if it's 1 for easier plotting
        if frames_to_plot.shape[0] == 1:
            frames_to_plot = frames_to_plot.squeeze(0)  # Shape becomes (frames, height, width)
    else:  # (T, H, W)
        frames_to_plot = video_tensor[start_frame:end_frame + 1, :, :]
    
    # Step 6: Plot the frames as a matrix
    num_frames_to_plot = frames_to_plot.shape[0]
    
    # Handle single frame case specially
    if num_frames_to_plot == 1:
        # Single frame: use a simple single subplot
        if figsize is None:
            figsize = (6, 6)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Set clipping values
        if clipping is None:
            global_min_plot = -0.003
            global_max_plot = 0.003
        else:
            global_min_plot, global_max_plot = clipping
        
        # Extract the single frame
        frame_data = frames_to_plot[0, :, :].numpy() - 1  # Subtract 1 as in original code
        im = ax.imshow(frame_data, cmap='hot', vmin=global_min_plot, vmax=global_max_plot)
        
        # Set title
        if real_frame_range is not None:
            real_start, real_end = real_frame_range
            tensor_total_frames = video_tensor.shape[1] if video_tensor.ndim == 4 else video_tensor.shape[0]
            frame_ratio = start_frame / (tensor_total_frames - 1) if tensor_total_frames > 1 else 0
            real_frame_idx = int(real_start + frame_ratio * (real_end - real_start))
            ax.set_title(f"Frame {real_frame_idx}", fontsize=12)
        else:
            ax.set_title(f"Frame {start_frame}", fontsize=12)
        
        ax.axis('off')
        fig.suptitle(title, fontsize=14)
        
        # Add colorbar for single frame
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.05)
        cbar.set_label('Intensity', rotation=270, labelpad=15)
        
        plt.tight_layout()
    else:
        # Multiple frames: use grid layout
        rows = (num_frames_to_plot + cols - 1) // cols  # Calculate the number of rows
        
        # Auto-calculate figsize if not provided
        if figsize is None:
            figsize = (cols * 1.5, rows * 1.5)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        # Handle axes array - flatten if it's a multi-dimensional array
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        # Set clipping values
        if clipping is None:
            global_min_plot = -0.003
            global_max_plot = 0.003
        else:
            global_min_plot, global_max_plot = clipping
        
        # Plot each frame in a subplot
        for i in range(num_frames_to_plot):
            ax = axes[i]
            # Use a heatmap colormap and apply consistent clipping
            # Convert tensor to numpy for plotting
            frame_data = frames_to_plot[i, :, :].numpy() - 1  # Subtract 1 as in original code
            im = ax.imshow(frame_data, cmap='hot', vmin=global_min_plot, vmax=global_max_plot)
            # Use real frame indices if provided, otherwise use relative indices
            if real_frame_range is not None:
                real_start, real_end = real_frame_range
                # Calculate the real frame index based on the selected range within the video tensor
                tensor_frame_idx = start_frame + i
                tensor_total_frames = video_tensor.shape[1] if video_tensor.ndim == 4 else video_tensor.shape[0]
                # Map tensor frame index to real frame index
                frame_ratio = tensor_frame_idx / (tensor_total_frames - 1) if tensor_total_frames > 1 else 0
                real_frame_idx = int(real_start + frame_ratio * (real_end - real_start))
                ax.set_title(f"Frame {real_frame_idx}", fontsize=8)
            else:
                ax.set_title(f"Frame {start_frame + i}", fontsize=8)
            ax.axis('off')
        
        # Hide any unused subplots
        for j in range(num_frames_to_plot, len(axes)):
            axes[j].axis('off')
        
        # Set title first
        fig.suptitle(title, fontsize=14)
        
        # Apply tight layout first to arrange subplots
        plt.tight_layout()
        
        # Add a colorbar positioned to the right without overlapping
        if len(axes) > 0:
            # Create colorbar using the entire grid of subplots with more aggressive padding
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, aspect=20, pad=0.05)
            cbar.set_label('Intensity', rotation=270, labelpad=15)
            
            # Force the colorbar to stay within the figure bounds
            cbar.ax.set_position([0.85, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        
        # Final adjustment to ensure no overlap
        plt.subplots_adjust(right=0.82)
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_spatial_dynamics(
    video_tensor: torch.Tensor,
    grid_rows: int = 10,
    grid_cols: int = 10,
    figsize: Tuple[float, float] = (15, 15),
    cmap: str = "hot",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Spatial Dynamics",
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    real_frame_range: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot spatial dynamics by showing average response over time for each spatial patch.
    
    This function divides the spatial dimensions into patches and shows the temporal
    dynamics (average response over time) for each patch in a grid layout.
    
    Args:
        video_tensor: Tensor of shape (C, T, H, W) or (T, H, W)
        grid_rows: Number of rows in the grid layout (default: 10)
        grid_cols: Number of columns in the grid layout (default: 10)
        figsize: Figure size (width, height)
        cmap: Colormap for visualization
        vmin, vmax: Color scale limits
        title: Plot title
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        real_frame_range: Tuple of (real_start, real_end) representing absolute frame indices from original dataset
        
    Returns:
        matplotlib Figure object
    """
    # Handle different tensor shapes
    if video_tensor.ndim == 4:  # (C, T, H, W)
        if video_tensor.shape[0] == 1:
            frames = video_tensor.squeeze(0)  # (T, H, W)
        else:
            frames = video_tensor.mean(dim=0)  # Average across channels
    elif video_tensor.ndim == 3:  # (T, H, W)
        frames = video_tensor
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {video_tensor.ndim}D")
    
    T, H, W = frames.shape
    
    # Derive patch size from grid dimensions
    patch_height = H // grid_rows
    patch_width = W // grid_cols
    
    # Calculate actual number of patches based on derived patch size
    patches_h = H // patch_height
    patches_w = W // patch_width
    total_patches = patches_h * patches_w
    
    # Ensure we don't exceed the requested grid size
    if total_patches > grid_rows * grid_cols:
        total_patches = grid_rows * grid_cols
    
    # Create figure
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    if grid_rows == 1:
        axes = axes.reshape(1, -1) if grid_cols > 1 else [axes]
    elif grid_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Calculate patch dynamics
    patch_dynamics = []
    patch_idx = 0
    for i in range(grid_rows):
        for j in range(grid_cols):
            if patch_idx >= total_patches:
                break
                
            # Extract patch coordinates using derived patch dimensions
            h_start = i * patch_height
            h_end = h_start + patch_height
            w_start = j * patch_width
            w_end = w_start + patch_width
            
            # Extract patch and compute average over spatial dimensions
            patch = frames[:, h_start:h_end, w_start:w_end]  # (T, patch_height, patch_width)
            patch_mean = patch.mean(dim=(1, 2))  # (T,) - average over spatial dimensions
            
            patch_dynamics.append(patch_mean)
            patch_idx += 1
    
    # Convert to tensor for easier manipulation
    patch_dynamics = torch.stack(patch_dynamics)  # (total_patches, T)
    
    # Determine color scale if not provided - FIXED VERSION
    if vmin is None or vmax is None:
        # Filter out NaN and Inf values before computing min/max
        valid_data = patch_dynamics[torch.isfinite(patch_dynamics)]
        if len(valid_data) == 0:
            # If all data is NaN/Inf, use default range
            vmin_auto, vmax_auto = -1.0, 1.0
        else:
            vmin_auto = valid_data.min().item()
            vmax_auto = valid_data.max().item()
        
        vmin = vmin if vmin is not None else vmin_auto
        vmax = vmax if vmax is not None else vmax_auto
    
    # Plot each patch dynamics
    for idx in range(total_patches):
        row = idx // grid_cols
        col = idx % grid_cols
        
        ax = axes[row, col]
        
        # Plot temporal dynamics
        # Use real frame indices if provided, otherwise use relative indices
        if real_frame_range is not None:
            real_start, real_end = real_frame_range
            time_points = torch.linspace(real_start, real_end, T)
        else:
            time_points = torch.arange(T)
        
        dynamics = patch_dynamics[idx].cpu().numpy()
        
        # Filter out NaN/Inf values for plotting
        finite_mask = np.isfinite(dynamics)
        if np.any(finite_mask):
            ax.plot(time_points[finite_mask], dynamics[finite_mask], linewidth=1.5, color='red')
            ax.fill_between(time_points[finite_mask], dynamics[finite_mask], alpha=0.3, color='red')
        else:
            # If all values are NaN/Inf, plot a flat line
            ax.axhline(y=0, color='red', linewidth=1.5)
        
        # Customize subplot
        ax.set_title(f'Patch {idx+1}', fontsize=8)
        ax.set_xlabel('Time', fontsize=6)
        ax.set_ylabel('Response', fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits for better visualization - FIXED VERSION
        if torch.isfinite(torch.tensor([vmin, vmax])).all():
            ax.set_ylim(vmin, vmax)
        else:
            # Use auto-scaling if limits are invalid
            ax.set_ylim(auto=True)
    
    # Hide unused subplots
    for idx in range(total_patches, grid_rows * grid_cols):
        row = idx // grid_cols
        col = idx % grid_cols
        axes[row, col].axis('off')
    
    # Set overall title and layout
    fig.suptitle(f'{title}\n({grid_rows}×{grid_cols} grid, {patch_height}×{patch_width} pixels per patch)', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spatial dynamics figure saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig
