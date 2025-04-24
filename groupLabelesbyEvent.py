import os
import shutil

# Define the source and target directories
source_dir = r"/UFO"
target_dir = r"/UFO_256_GroupedByEvents"

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Iterate over all files in the source directory
for filename in os.listdir(source_dir):
    if filename.lower().endswith(('.tif', '.tiff')):  # Check if the file is a GeoTIFF
        # Extract the first three letters of the filename
        prefix = filename[:3]
        # Construct the path to the target folder
        target_folder = os.path.join(target_dir, prefix)
        # Ensure the target folder exists
        os.makedirs(target_folder, exist_ok=True)
        # Construct the full source and target paths
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_folder, filename)
        # Copy the file
        shutil.copy(source_path, target_path)

print("Files have been organized and copied successfully.")
