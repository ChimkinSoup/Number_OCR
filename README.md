# Number_OCR
A neural network built from scratch using only NumPy and linear algebra (no PyTorch or TensorFlow). You can train your own model or load the included pre-trained model to classify your own hand-drawn digits

## Features
- 2-layer neural network
- Trained on MNIST + 400k augmented MNIST images
- Adam optimizer
- Leaky ReLU activation
- Weight decay (L2 regularization)
- One-hot encoding
- Early stopping
- Interactive drawing canvas
- Real-time 28x28 preview of what the model sees
- Keyboard shortcuts
- Includes a pre-trained model achieving **99.07% accuracy** on MNIST test data

## Running It
1. Clone this repository
```bash
git clone https://github.com/ChimkinSoup/Number_OCR.git
cd Number_OCR
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python src/main.py
```

Inside `src/main.py` you can toggle:
- `TRAIN_LOCALLY = False` (default) to use the included pre-trained model
- `TRAIN_LOCALLY = True` to train your own model

## Pre-trained Model
- A pre-trained network (`DefaultModel.npz`) is included in the repository
- It was trained on **MNIST + augmented data** and reaches **99.07% accuracy** on MNIST test data
- This model loads automatically when `TRAIN_LOCALLY = False`

## Using The Augmented Dataset (Optional)
If you want to retrain the model using both MNIST and additional augmented data (by default the model trains on only the MNIST dataset):

1. Download the augmented MNIST dataset from: <https://drive.google.com/drive/folders/1OqtQUWKZQFkyAt5nbJDyyocfgMjibDRH?usp=drive_link> 
2. Extract the dataset and ensure it's titled `Augmented_MNIST_IDX`. Place it under `Data`. The structure should look like:
```
Data/
    Augmented_MNIST_IDX/
        augmented_mnist_labels.idx1-ubyte
        augmented_mnist_images.idx3-ubyte
```
3. Set `TRAIN_LOCALLY = True` in `src/main.py`.

## Data Source

This project uses augmented MNIST images provided by **Alexandre Le Mercier**:  
https://www.kaggle.com/ds/6967763  
Dataset title: *400k Augmented MNIST: Extended Handwritten Digits*.

Data © Alexandre Le Mercier (2025), available on Kaggle.