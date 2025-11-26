import numpy as np
from PIL import Image
import io
from pathlib import Path
        
def byte_to_int(byte):
    return int.from_bytes(byte, 'big')


    


def read_images(filename, max_images=None):
    if not filename.exists():
        raise FileNotFoundError(f"MNIST data file not found: {filename}")
    
    all_images = []
    try:
        with open(filename, 'rb') as file:
            _ = file.read(4)  # Magic number
            number_images = byte_to_int(file.read(4))
            if max_images:
                number_images = min(number_images, max_images)
            number_rows = byte_to_int(file.read(4))
            number_columns = byte_to_int(file.read(4))
            buffered_data = np.frombuffer(
                file.read(number_images * number_columns * number_rows), 
                dtype=np.uint8
            )
            all_images = buffered_data.reshape(number_images, number_columns * number_rows)
        return (all_images.astype(np.float32) / 255.0).T # Scaling pixel values to be within [0, 1] 
    except Exception as e:
        raise ValueError(f"Error reading image file {filename}: {e}")

def read_labels(filename, max_labels=None):
    if not filename.exists():
        raise FileNotFoundError(f"MNIST data file not found: {filename}")
    
    all_labels = []
    try:
        with open(filename, 'rb') as file:
            _ = file.read(4)  # Magic number
            number_labels = byte_to_int(file.read(4))
            if max_labels:
                number_labels = min(number_labels, max_labels)

            # Read all labels at once for better performance
            all_labels = np.frombuffer(file.read(number_labels), dtype=np.uint8)
        return all_labels
    except Exception as e:
        raise ValueError(f"Error reading label file {filename}: {e}") 

def shuffleData(images, labels):
    # images is in (features, samples) format, so use shape[1] for number of samples
    num_samples = images.shape[1]
    shuffledIndex = np.random.permutation(num_samples)
    return images[:, shuffledIndex], labels[shuffledIndex]
