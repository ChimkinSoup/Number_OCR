# MNIST_Neural_Network
Neural network created without PyTorch or TensorFlow, only linear algebra. Personally train your own model, or load a pre-existing model to guess your own hand-drawn numbers.

## Features
- Leaky ReLu
- Weight decay
- One-hot encoding
- Early stopping
- Adam optimizer
- Trained with the MNIST dataset, and 400k augmented images based off MNIST
- 2-layer neural network 
- Interactive drawing canvas
- Real-time 28x28 preview of what the model sees
- Keyboard shortcuts
- Achieved 99.07% accuracy on MNIST testing data

## Running It
1. Clone this repository:
<insert necessary code>

2. Install dependencies:
<insert necessary code>

3. Run the application:
<code>
Set TRAIN_LOCALLY to True if you want to train your own model (by default the program uses a pre-determined network)
- (Optional) Go to <link> and download the Augmented_MNIST_IDX folder. Model will be trained on 60k MNIST images + 400k augmented images (more accuracy, slightly longer training times)

## Data Source

This project uses augmented MNIST images provided by **Alexandre Le Mercier**:  
https://www.kaggle.com/ds/6967763  
Dataset title: *400k Augmented MNIST: Extended Handwritten Digits*.

Data © Alexandre Le Mercier (2025), available on Kaggle.
