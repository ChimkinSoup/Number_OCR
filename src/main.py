import numpy as np
import pandas as pd
from pathlib import Path
import fileReading as reader
import NeuralNetwork
import draw

# =-=-=-=-=-=-=-=-=-=-=-=-=
TRAIN_LOCALLY = False # Set to True to train the model locally
# =-=-=-=-=-=-=-=-=-=-=-=-=

BASE_DIRECTORY = Path(__file__).parent.parent  # Go up to project root
TEST_LABELS_FILENAME = BASE_DIRECTORY / 'Data' / 'MNIST' / 't10k-labels.idx1-ubyte'
TEST_IMAGES_FILENAME = BASE_DIRECTORY / 'Data' / 'MNIST' / 't10k-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = BASE_DIRECTORY / 'Data' / 'MNIST' / 'train-labels.idx1-ubyte'
TRAIN_IMAGES_FILENAME = BASE_DIRECTORY / 'Data' / 'MNIST' / 'train-images.idx3-ubyte'
AUGMENTED_IMAGE_FILENAME = BASE_DIRECTORY / 'Data' / 'Augmented_MNIST_IDX' / 'augmented_mnist_images.idx3-ubyte'
AUGMENTED_LABEL_FILENAME = BASE_DIRECTORY / 'Data' / 'Augmented_MNIST_IDX' / 'augmented_mnist_labels.idx1-ubyte'


def main():
    print("Number OCR - Starting application...")

    if TRAIN_LOCALLY:
        print("Training mode enabled (Might take a while to train)")
        
        try:
            XVal = reader.read_images(TEST_IMAGES_FILENAME)
            yVal = reader.read_labels(TEST_LABELS_FILENAME)
            XTrain = reader.read_images(TRAIN_IMAGES_FILENAME)
            yTrain = reader.read_labels(TRAIN_LABELS_FILENAME)
        except FileNotFoundError as e:
            print(f"Error loading standard MNIST data: {e}")
            return

        XAugmented = None
        yAugmented = None

        # Check if augmented data exists
        if AUGMENTED_IMAGE_FILENAME.exists() and AUGMENTED_LABEL_FILENAME.exists():
            print("Augmented data found. Training on mixed dataset.")
            try:
                XAugmented = reader.read_images(AUGMENTED_IMAGE_FILENAME)
                yAugmented = reader.read_labels(AUGMENTED_LABEL_FILENAME)
            except Exception as e:
                print(f"Error loading augmented data: {e}")
                print("Falling back to regular MNIST data only.")
                XAugmented = None
                yAugmented = None
        else:
            print("Augmented data NOT found. Training on regular MNIST dataset only.")

        test = NeuralNetwork.NeuralNetwork()
        test.trainAugmented(XTrain, yTrain, XAugmented, yAugmented, XVal, yVal)
        test.saveData()
        print("Model's Testing Accuracy: ", test.bestAccuracy)
        print("Took ", test.epochCount, " epochs to train")
        
        # Set the cached model to the one we just trained
        NeuralNetwork.cached_model = test
        print("Opening drawing application with the newly trained model...")
        draw.main()
    else:
        print("Opening drawing application with the pre-trained model...")
        draw.main()





if __name__ == "__main__":
    main()
