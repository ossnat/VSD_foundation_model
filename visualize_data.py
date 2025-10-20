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
        "shuffle": True, # Set to True to shuffle the data
        "window_size": 0
    }

    # Use the load_dataset function to create a DataLoader
    vsd_dataloader = load_dataset(cfg)

    print("DataLoader created successfully.")

    # Print dataset stats: total samples and frames per sample
    try:
        total_samples = len(vsd_dataloader.dataset)
        # Safely inspect the first sample to infer frames per sample
        if total_samples > 0:
            first_sample = vsd_dataloader.dataset[0]
            if isinstance(first_sample, dict) and 'video' in first_sample:
                frames_per_sample = first_sample['video'].shape[1]
                start_frame = first_sample.get('start_frame', None)
                end_frame = first_sample.get('end_frame', None)
                print(f"Total samples: {total_samples}; Frames per sample: {frames_per_sample}")
            else:
                print(f"Total samples: {total_samples}; unable to infer frames per sample")
        else:
            print("Total samples: 0")
    except Exception as e:
        print(f"Could not compute dataset stats: {e}")

    # Get the first batch from the DataLoader and print its shape
    try:
        batch = next(iter(vsd_dataloader))
        print(f"Shape of the first batch (video tensor): {batch['video'].shape}")
        if 'start_frame' in batch and 'end_frame' in batch:
            start_frames = batch['start_frame'].tolist()
            end_frames = batch['end_frame'].tolist()
            frame_ranges = [(start, end) for start, end in zip(start_frames, end_frames)]
            print(f"First batch frame ranges: {frame_ranges}")
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
                'mask': batch['mask'][random_index_in_batch] if 'mask' in batch and batch['mask'] is not None else None,
                'start_frame': batch['start_frame'][random_index_in_batch] if 'start_frame' in batch else None,
                'end_frame': batch['end_frame'][random_index_in_batch] if 'end_frame' in batch else None
            }
            video_tensor = sample['video']

            print(f"Successfully pulled a random sample (index {random_index_in_batch}) from a batch.")
            print(f"Shape of the selected sample video tensor: {video_tensor.shape}")
            if sample.get('start_frame') is not None and sample.get('end_frame') is not None:
                start_frame = sample['start_frame'].item() if hasattr(sample['start_frame'], 'item') else sample['start_frame']
                end_frame = sample['end_frame'].item() if hasattr(sample['end_frame'], 'item') else sample['end_frame']
                print(f"Selected sample absolute frame range: ({start_frame}, {end_frame})")

            # Use the new plot_frames_sequence function to replace steps 4-6
            real_frame_range = None
            if sample.get('start_frame') is not None and sample.get('end_frame') is not None:
                start_frame_val = sample['start_frame'].item() if hasattr(sample['start_frame'], 'item') else sample['start_frame']
                end_frame_val = sample['end_frame'].item() if hasattr(sample['end_frame'], 'item') else sample['end_frame']
                real_frame_range = (start_frame_val, end_frame_val)
            
            fig1 = plot_frames_sequence(
                video_tensor=video_tensor,
                start_frame=27,
                end_frame=68,
                clipping=(-0.003, 0.003),
                cols=10,
                title=f"Random Sample {random_index_in_batch} - Frame Sequence (frames {sample.get('start_frame', '?')}-{sample.get('end_frame', '?')})",
                show_plot=False,
                real_frame_range=real_frame_range
            )

            # Plot spatial dynamics
            print(f"\n📊 Plotting spatial dynamics for sample {random_index_in_batch}...")
            fig2 = plot_spatial_dynamics(
                video_tensor=video_tensor,
                grid_rows=10,
                grid_cols=10,
                title=f"Random Sample {random_index_in_batch} - Spatial Dynamics (frames {sample.get('start_frame', '?')}-{sample.get('end_frame', '?')})",
                show_plot=False,
                real_frame_range=real_frame_range
            )
            
            # Show both plots
            print("Displaying both plots...")
            import matplotlib.pyplot as plt
            
            # Display both figures
            plt.figure(fig1.number)
            plt.show(block=False)
            plt.figure(fig2.number)
            plt.show(block=False)
            
            # Keep the script running to display plots
            input("Press Enter to close plots and continue...")

        except StopIteration:
            print("Error: The DataLoader is empty.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


