# ==================================
# File: src/utils/visualization.py
# ==================================

import torch
import torchvision.utils as vutils
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
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot a sequence of video frames in a grid layout.
    
    This function replicates steps 4-6 from the original visualization code:
    4. Define the frame range to extract for plotting
    5. Extract the desired frame range
    6. Plot the frames as a matrix
    
    Args:
        video_tensor: Video tensor of shape (C, T, H, W) or (T, H, W)
        start_frame: Starting frame index (default: 0)
        end_frame: Ending frame index (if None, uses all frames from start_frame)
        clipping: Tuple of (min, max) values for color scaling (default: (-0.003, 0.003))
        cols: Number of columns in the grid (default: 10)
        figsize: Figure size as (width, height) (default: auto-calculated)
        title: Title for the plot (default: "Video Frames Sequence")
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot (default: True)
        
    Returns:
        matplotlib Figure object
    """
    # Step 4: Define the frame range to extract for plotting
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
    
    # Step 5: Extract the desired frame range
    if video_tensor.ndim == 4:  # (C, T, H, W)
        frames_to_plot = video_tensor[:, start_frame:end_frame + 1, :, :]
        # Remove the channel dimension if it's 1 for easier plotting
        if frames_to_plot.shape[0] == 1:
            frames_to_plot = frames_to_plot.squeeze(0)  # Shape becomes (frames, height, width)
    else:  # (T, H, W)
        frames_to_plot = video_tensor[start_frame:end_frame + 1, :, :]
    
    # Step 6: Plot the frames as a matrix
    num_frames_to_plot = frames_to_plot.shape[0]
    rows = (num_frames_to_plot + cols - 1) // cols  # Calculate the number of rows
    
    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (cols * 1.5, rows * 1.5)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration
    
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
