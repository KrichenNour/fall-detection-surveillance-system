# Fall Detection Surveillance System

A deep learning-based fall detection system using a two-stream transformer model that processes foreground masks and optical flow for accurate fall detection in surveillance videos.

## Features

- **Preprocessing**: Extracts foreground masks and optical flow from video sequences using background subtraction and optical flow algorithms.
- **Two-Stream Transformer Model**: Utilizes a transformer-based architecture with two streams for processing spatial (foreground masks) and temporal (optical flow) information.
- **Training and Evaluation**: Includes scripts for training the model, validation, and performance metrics.
- **Data Augmentation**: Supports data augmentation techniques like horizontal flipping and temporal jittering during training.

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- tqdm

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/KrichenNour/fall-detection-surveillance-system.git
   cd fall-detection-surveillance-system
   ```

2. Install the required packages:
   ```bash
   pip install torch torchvision opencv-python numpy pandas matplotlib scikit-learn tqdm
   ```

## Usage

### Data Preparation

1. Place your video data in the `Processed_Data/Fall` and `Processed_Data/No_Fall` directories.

2. Run the preprocessing script to extract foreground masks and optical flow:
   ```bash
   python preprocessing.py Fall
   python preprocessing.py No_Fall
   ```

   This will generate `.npy` files in `Processed_For_DL/Fall/fg`, `Processed_For_DL/Fall/flow`, etc.

3. Create a `labels.csv` file with columns `filename` and `label` (0 for No Fall, 1 for Fall).

### Training the Model

Run the training script:
```bash
python Two_Stream_Transformer.py
```

This will train the model and save the best model weights to `best_two_stream_transformer_aug_3.pth`.

### Testing

Use the trained model for inference on new data. Modify the test script as needed.

## Dataset

The system is designed to work with video datasets for fall detection. The preprocessing expects videos in MP4 format. The model has been trained and evaluated on a dataset with fall and no-fall sequences.

## Model Architecture

The Two-Stream Transformer consists of:
- **Patch Embedding**: Converts image patches into embeddings.
- **Positional Encoding**: Adds positional information to the sequence.
- **Transformer Encoder**: Processes the sequence with multi-head attention.
- **Classifier Head**: A simple MLP for binary classification.

## Results
Training history and additional metrics are saved in the `training_results/` directory.

