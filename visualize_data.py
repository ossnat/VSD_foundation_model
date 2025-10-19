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
from src.utils.visualization import plot_frames_sequence, plot_spatial_dynamics


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
        "normalize": True,
        "normalization_type": "baseline_zscore",
        "baseline_frame": 20,
        "frame_start": 1,
        "frame_end": 100,
        "batch_size": 4, # You can adjust the batch size
        "num_workers": 2, # You can adjust the number of workers
        "shuffle": True # Set to True to shuffle the data
    }

    # Use the load_dataset function to create a DataLoader
    vsd_dataloader = load_dataset(cfg)

    print("DataLoader created successfully.")

    # Get the first batch from the DataLoader and print its shape
    try:
        batch = next(iter(vsd_dataloader))
        print(f"Shape of the first batch (video tensor): {batch['video'].shape}")
        # If you also have a mask, you can print its shape as well
        if 'mask' in batch and batch['mask'] is not None:
            print(f"Shape of the first batch (mask tensor): {batch['mask'].shape}")

    except StopIteration:
        print("The DataLoader is empty.")
    except Exception as e:
        print(f"An error occurred while getting the first batch: {e}")

    if 'vsd_dataloader' not in locals():
        print("Error: vsd_dataloader is not defined. Please run the cell that creates the DataLoader first.")
    else:
        try:
            # Use the batch we already got above (no need to get it again)

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

            # Use the new plot_frames_sequence function to replace steps 4-6
            plot_frames_sequence(
                video_tensor=video_tensor,
                start_frame=27,
                end_frame=68,
                clipping=(-0.003, 0.003),
                cols=10,
                title=f"Random Sample {random_index_in_batch} - Frame Sequence",
                show_plot=True
            )

            # Plot spatial dynamics
            print(f"\n📊 Plotting spatial dynamics for sample {random_index_in_batch}...")
            plot_spatial_dynamics(
                video_tensor=video_tensor,
                grid_rows=10,
                grid_cols=10,
                title=f"Random Sample {random_index_in_batch} - Spatial Dynamics",
                show_plot=True
            )

        except StopIteration:
            print("Error: The DataLoader is empty.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


