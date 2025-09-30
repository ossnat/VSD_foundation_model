import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random # Import random for selecting a random sample

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import useful symbols from src subpackages
from src.data import *  # noqa: F401,F403
# from src.models import *  # noqa: F401,F403
# from src.preprocessing import *  # noqa: F401,F403
# from src.training import *  # noqa: F401,F403
from src.utils import *  # noqa: F401,F403


def load_hdf5(hdf5_path: Path):
    import h5py

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    f = h5py.File(str(hdf5_path), mode="r")
    return f


def main():
    # Google Drive for desktop typical path
    hdf5_path = Path(r"G:\My Drive\HDF5_DATA_AFTER_PREPROCESSING2\vsd_video_data.hdf5")

    try:
        h5 = load_hdf5(hdf5_path)
    except Exception as e:
        print(f"Failed to open HDF5: {e}")
        return

    # Simple introspection
    def _printer(name, obj):
        try:
            shape = getattr(obj, 'shape', None)
            dtype = getattr(obj, 'dtype', None)
            if shape is not None:
                print(f"dataset: {name}, shape={shape}, dtype={dtype}")
        except Exception:
            pass

    try:
        h5.visititems(_printer)
    finally:
        h5.close()

    # Define a configuration dictionary for the VSD dataset
    # Make sure the 'vsd_hdf5_path' points to your HDF5 file
    cfg = {
        "dataset": "vsd",
        "vsd_hdf5_path": Path(r"G:\My Drive\HDF5_DATA_AFTER_PREPROCESSING2\vsd_video_data.hdf5"),
        "batch_size": 4, # You can adjust the batch size
        "num_workers": 2, # You can adjust the number of workers
        "shuffle": True # Set to True to shuffle the data
    }

    # Use the load_dataset function to create a DataLoader
    vsd_dataloader = load_dataset(cfg)

    print("DataLoader created successfully.")

    # Get the first batch from the DataLoader and print its shape
    try:
        first_batch = next(iter(vsd_dataloader))
        print(f"Shape of the first batch (video tensor): {first_batch['video'].shape}")
        # If you also have a mask, you can print its shape as well
        if 'mask' in first_batch and first_batch['mask'] is not None:
            print(f"Shape of the first batch (mask tensor): {first_batch['mask'].shape}")

    except StopIteration:
        print("The DataLoader is empty.")
    except Exception as e:
        print(f"An error occurred while getting the first batch: {e}")

    if 'vsd_dataloader' not in locals():
        print("Error: vsd_dataloader is not defined. Please run the cell that creates the DataLoader first.")
    else:
        try:
            # 1. Get a batch from the DataLoader
            # We use iter() and next() to get one batch
            batch = next(iter(vsd_dataloader))

            # 2. Select a random sample index from the batch
            batch_size = batch['video'].shape[0]
            random_index_in_batch = random.randint(0, batch_size - 1)

            # 3. Access the random sample from the batch
            sample = {
                'video': batch['video'][random_index_in_batch],
                'mask': batch['mask'][random_index_in_batch] if 'mask' in batch and batch['mask'] is not None else None
            }
            video_tensor = sample['video']

            print(f"Successfully pulled a random sample (index {random_index_in_batch}) from a batch.")
            print(f"Shape of the selected sample video tensor: {video_tensor.shape}")


            # 4. Define the frame range to extract for plotting
            start_frame_plot = 27
            end_frame_plot = 68

            # Ensure the frame range is within the bounds of the video tensor
            if start_frame_plot < 0 or end_frame_plot >= video_tensor.shape[1] or start_frame_plot > end_frame_plot:
                print(f"Warning: The requested frame range [{start_frame_plot}:{end_frame_plot}] is out of bounds for the video tensor with {video_tensor.shape[1]} frames. Plotting all frames.")
                start_frame_plot = 0
                end_frame_plot = video_tensor.shape[1] - 1


            # 5. Extract the desired frame range
            # The tensor shape is (channels, frames, height, width)
            # We need to slice along the frames dimension (dimension 1)
            frames_to_plot = video_tensor[:, start_frame_plot : end_frame_plot + 1, :, :]

            # Remove the channel dimension if it's 1 for easier plotting
            if frames_to_plot.shape[0] == 1:
                frames_to_plot = frames_to_plot.squeeze(0) # Shape becomes (frames, height, width)


            # 6. Plot the frames as a matrix
            num_frames_to_plot = frames_to_plot.shape[0]
            cols = 10  # Number of columns in the grid, adjust as needed
            rows = (num_frames_to_plot + cols - 1) // cols # Calculate the number of rows

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5)) # Adjust figsize as needed
            axes = axes.flatten() # Flatten the 2D array of axes for easier iteration

            # Determine global min and max for consistent clipping from the extracted frames
            global_min_plot = -0.003
            global_max_plot = 0.003


            # Plot each frame in a subplot
            for i in range(num_frames_to_plot):
                ax = axes[i]
                # Use a heatmap colormap (e.g., 'hot', 'viridis', 'plasma')
                # Apply consistent clipping using vmin and vmax
                # Convert tensor to numpy for plotting
                im = ax.imshow(frames_to_plot[i, :, :].numpy()-1, cmap='hot', vmin=global_min_plot, vmax=global_max_plot)
                ax.set_title(f"Frame {start_frame_plot + i}", fontsize=8)
                ax.axis('off')

            # Hide any unused subplots
            for j in range(num_frames_to_plot, len(axes)):
                axes[j].axis('off')

            # Add a colorbar
            fig.colorbar(im, ax=axes.ravel().tolist())


            plt.tight_layout() # Adjust layout to prevent titles overlapping
            plt.show()

            plt.close('all') # Close all plot figures after displaying the matrix

        except StopIteration:
            print("Error: The DataLoader is empty.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


