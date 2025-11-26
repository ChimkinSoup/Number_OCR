import numpy as np
from pathlib import Path
from PIL import Image
import struct

BASE_DIRECTORY = Path(__file__).parent.parent  # Go up to project root
AUGMENTED_DATA_PATH = BASE_DIRECTORY / 'Data' / 'Augmented_MNIST'


def create_idx_dataset(base_folder, output_image_file, output_label_file, image_size=(28, 28)):
    """
    Convert JPG images from Augmented_MNIST folders to IDX format.
    
    Args:
        base_folder: Path to the Augmented_MNIST directory containing Label_* folders
        output_image_file: Path where the IDX images file will be saved
        output_label_file: Path where the IDX labels file will be saved
        image_size: Tuple of (height, width) for resizing images
    """
    base_folder = Path(base_folder)
    all_images = []
    all_labels = []

    if not base_folder.exists():
        raise FileNotFoundError(f"Directory not found: {base_folder}")

    # Loop over each label folder (Label_0, Label_1, ..., Label_9 or label_0, label_1, ...)
    # Get all possible label folders and deduplicate (Windows filesystem is case-insensitive)
    label_folders_list = list(base_folder.glob("Label_*")) + list(base_folder.glob("label_*"))
    # Deduplicate by using a dict with normalized path as key (handle case-insensitive filesystems)
    unique_folders = {}
    for folder in label_folders_list:
        # Use lowercase path as key to deduplicate
        key = str(folder).lower()
        if key not in unique_folders:
            unique_folders[key] = folder
    label_folders = sorted(unique_folders.values(), key=lambda p: p.name.lower())
    
    if len(label_folders) == 0:
        raise ValueError(f"No Label_* or label_* folders found in {base_folder}")
    
    print(f"Found {len(label_folders)} unique label folders")
    
    total_images_processed = 0
    
    for label_folder in label_folders:
        # Extract label from folder name (handle both "Label_X" and "label_X")
        folder_name = label_folder.name
        if folder_name.lower().startswith("label_"):
            label = int(folder_name.split("_")[1])
        else:
            print(f"Warning: Skipping unrecognized folder format: {folder_name}")
            continue
        
        print(f"Processing {folder_name} (label {label})...")
        
        # Read all JPG images in this label folder
        jpg_files = sorted(label_folder.glob("*.jpg"))
        print(f"  Found {len(jpg_files)} images")
        
        for idx, img_file in enumerate(jpg_files, 1):
            try:
                img = Image.open(img_file).convert('L')  # Convert to grayscale
                img = img.resize(image_size)              # Resize to 28x28
                img_array = np.array(img, dtype=np.uint8).flatten()
                all_images.append(img_array)
                all_labels.append(label)
                total_images_processed += 1
                
                # Print progress every 1000 images
                if total_images_processed % 1000 == 0:
                    print(f"  Progress: {total_images_processed} images processed...")
            except Exception as e:
                print(f"Warning: Could not read {img_file}: {e}")
                continue
    
    if len(all_images) == 0:
        raise ValueError("No images were successfully loaded!")
    
    all_images = np.array(all_images, dtype=np.uint8)
    all_labels = np.array(all_labels, dtype=np.uint8)
    
    print(f"\nTotal images processed: {len(all_images)}")
    print(f"Total labels: {len(all_labels)}")
    print(f"Image shape: {image_size}")

    # Ensure output directory exists
    output_image_file = Path(output_image_file)
    output_label_file = Path(output_label_file)
    output_image_file.parent.mkdir(parents=True, exist_ok=True)
    output_label_file.parent.mkdir(parents=True, exist_ok=True)

    # Write combined IDX images file
    print(f"\nWriting images to {output_image_file}...")
    with open(output_image_file, 'wb') as f:
        # Magic number (0x00000803), number of images, rows, columns
        f.write(struct.pack('>IIII', 0x00000803, len(all_images), image_size[0], image_size[1]))
        f.write(all_images.tobytes())

    # Write combined IDX labels file
    print(f"Writing labels to {output_label_file}...")
    with open(output_label_file, 'wb') as f:
        # Magic number (0x00000801), number of labels
        f.write(struct.pack('>II', 0x00000801, len(all_labels)))
        f.write(all_labels.tobytes())

    print(f"\n✓ IDX files created successfully!")
    print(f"  Images: {output_image_file}")
    print(f"  Labels: {output_label_file}")


def main():
    """Main function to convert Augmented_MNIST JPGs to IDX format."""
    print("=" * 60)
    print("Converting Augmented_MNIST JPG images to IDX format")
    print("=" * 60)
    
    # Define output paths
    output_dir = BASE_DIRECTORY / 'Data' / 'Augmented_MNIST_IDX'
    output_image_file = output_dir / 'augmented_mnist_images.idx3-ubyte'
    output_label_file = output_dir / 'augmented_mnist_labels.idx1-ubyte'
    
    try:
        create_idx_dataset(
            base_folder=AUGMENTED_DATA_PATH,
            output_image_file=output_image_file,
            output_label_file=output_label_file,
            image_size=(28, 28)
        )
        print("\n" + "=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()