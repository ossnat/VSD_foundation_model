import os
import numpy as np
import scipy.io
import h5py
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


def convert_mat_to_hdf5(input_main_dir, output_hdf5_path, filename_pattern, exclude_keys=None):
    """
    Converts video data from .mat files matching a filename pattern within
    subdirectories of a main directory into a single HDF5 file, excluding
    specified keys.

    Args:
        input_main_dir (str): Path to the main directory containing subdirectories
                              with .mat files.
        output_hdf5_path (str): Path where the HDF5 file will be created.
        filename_pattern (str): The pattern to match for .mat filenames (e.g., 'condsXn.mat').
        exclude_keys (list, optional): A list of keys to exclude from the .mat files.
                                       Defaults to None (include all keys).
    # # --- Example Usage ---
        input_directory = 'preprocessed_VSDdata'
        output_hdf5_store = 'HDF5_DATA/vsd_video_data.hdf5' # Example: '/content/drive/MyDrive/VSD_FM/vsd_video_data.hdf5'
        filename_pattern = 'condsXA.mat'
        keys_to_exclude = ['blankAN'] # Example list of keys to exclude

        # # # Run the conversion
        convert_mat_to_hdf5(input_directory, output_hdf5_store, filename_pattern, exclude_keys=keys_to_exclude)

    """
    print(f"Starting conversion from '{input_main_dir}' to HDF5 file at '{output_hdf5_path}'")

    if exclude_keys is None:
        exclude_keys = []

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_hdf5_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Open the HDF5 file in write mode ('w')
    with h5py.File(output_hdf5_path, 'w') as hf:
        print(f"Created HDF5 file: {output_hdf5_path}")

        # Iterate through subdirectories
        for subdir, _, files in os.walk(input_main_dir):
            for file in files:
                if file == filename_pattern: # Match the specific filename pattern
                    mat_path = os.path.join(subdir, file)
                    relative_subdir = os.path.relpath(subdir, input_main_dir)
                    hdf5_group_name = relative_subdir.replace(os.sep, '_') # Use subdir name as group label

                    print(f"Processing file: {mat_path}")

                    try:
                        # Load the .mat file content
                        mat_content = scipy.io.loadmat(mat_path)

                        # Create a group in the HDF5 file for this subdirectory
                        group = hf.create_group(hdf5_group_name)
                        print(f"Created HDF5 group: /{hdf5_group_name}")

                        # Iterate through keys in the loaded .mat content
                        for key, data in mat_content.items():
                            # Exclude standard .mat header keys and specified exclude_keys
                            if not key.startswith('__') and key not in exclude_keys:
                                print(f"  Adding dataset '{key}' to group /{hdf5_group_name}")
                                try:
                                    # Convert data to a numpy array if it's not already
                                    if not isinstance(data, np.ndarray):
                                        data = np.array(data)

                                    # Write the data to the HDF5 group
                                    group.create_dataset(key, data=data)
                                except Exception as e:
                                    print(f"    Error writing dataset '{key}': {e}. Skipping.")
                            else:
                                print(f"  Skipping key '{key}' (either excluded or header)")

                    except Exception as e:
                        print(f"Error processing {mat_path}: {e}. Skipping file.")

    print(f"\nConversion finished. Data written to HDF5 file at '{output_hdf5_path}'.")


def test_view_sample_hdf5_converted_file(hdf5_file_path = 'HDF5_DATA/vsd_video_data.hdf5'):

    global item
    print(f"Attempting to open HDF5 file: {hdf5_file_path}")
    if not os.path.exists(hdf5_file_path):
        print(f"Error: HDF5 file not found at {hdf5_file_path}. Please run the conversion function first.")
    else:
        try:
            # Open the HDF5 file in read mode ('r')
            with h5py.File(hdf5_file_path, 'r') as hf:
                print("\nSuccessfully opened HDF5 file.")
                print("Contents of the HDF5 file:")

                # List the top-level items (groups)
                top_level_items = list(hf.keys())
                print(f"Top-level items (groups): {top_level_items}")

                if top_level_items:
                    # Pick the first group to explore its contents
                    first_group_name = top_level_items[2]
                    first_group = hf[first_group_name]
                    print(f"\nContents of the first group ('/{first_group_name}'):")

                    # List items within the first group (datasets)
                    group_items = list(first_group.keys())
                    print(f"Datasets in '/{first_group_name}': {group_items}")

                    # Find a dataset that looks like video data (assuming it has at least 3 dimensions)
                    video_dataset_name = None
                    for item_name in group_items:
                        item = first_group[item_name]
                        if isinstance(item, h5py.Dataset) and item.ndim >= 3:
                            video_dataset_name = item_name
                            print(f"Found potential video dataset: '{video_dataset_name}' with shape {item.shape}")
                            break  # Found a candidate, let's use this one

                    if video_dataset_name:
                        video_data_dataset = first_group[video_dataset_name]
                        dataset_shape = video_data_dataset.shape
                        print(
                            f"\nAccessing dataset: '/{first_group_name}/{video_dataset_name}' with shape {dataset_shape}")

                        # Assuming the shape is (pixels, frames, trials) based on the original .mat structure
                        # We need to select a random trial and a random frame within that trial.
                        if dataset_shape[2] > 0 and dataset_shape[1] > 0:  # Check if there are trials and frames
                            random_trial_index = 2  # random.randint(0, dataset_shape[2] - 1)
                            random_frame_index = 22  # random.randint(0, dataset_shape[1] - 1)

                            print(f"Selecting random trial index: {random_trial_index}")
                            print(f"Selecting random frame index: {random_frame_index}")

                            # Extract the data for the random trial and frame
                            # The slice needs to be (pixels, specific_frame, specific_trial)
                            try:
                                frame_data_flat = video_data_dataset[:, random_frame_index, random_trial_index]

                                # Assuming the pixels dimension corresponds to (height * width) and height=width=100
                                expected_pixels = 100 * 100
                                if frame_data_flat.shape[0] == expected_pixels:
                                    height = 100
                                    width = 100
                                    # Reshape the flat frame data into a 2D image
                                    frame_image = frame_data_flat.reshape((height, width))

                                    print(
                                        f"\nPlotting random frame from trial {random_trial_index}, frame {random_frame_index}:")

                                    # Plot the frame
                                    plt.figure(figsize=(6, 6))
                                    frame_image_norm = (frame_image - np.min(frame_image)) / (
                                                np.max(frame_image) - np.min(frame_image))
                                    plt.imshow(frame_image_norm,
                                               cmap='viridis')  # Assuming grayscale video, adjust cmap if needed
                                    plt.title(
                                        f'Random Frame from /{first_group_name}/{video_dataset_name}\nTrial: {random_trial_index}, Frame: {random_frame_index}')
                                    plt.colorbar(label='Pixel Intensity')
                                    plt.axis('off')  # Hide axes ticks
                                    plt.show()
                                else:
                                    print(
                                        f"Error: Unexpected number of pixels in the selected frame data ({frame_data_flat.shape[0]}). Expected {expected_pixels}. Cannot reshape and plot.")

                            except Exception as e:
                                print(f"Error extracting or plotting frame data: {e}")

                        else:
                            print("Skipping frame plotting: Dataset has no trials or frames.")

                    else:
                        print(
                            "No suitable video dataset (with >= 3 dimensions) found in the first group to plot a frame.")

                else:
                    print("No top-level items (groups) found in the HDF5 file.")

        except Exception as e:
            print(f"An error occurred while trying to open or read the HDF5 file: {e}")


