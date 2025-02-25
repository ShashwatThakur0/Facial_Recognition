# Facial Emotion Recognition System

A real-time facial emotion recognition system that detects and classifies emotions into three categories: Positive, Negative, and Neutral. The system uses computer vision and deep learning techniques to analyze facial expressions in real-time through your webcam.

## Features

- Real-time facial emotion detection using webcam
- Three emotion categories: Positive, Negative, and Neutral
- Enhanced face detection with 40% padding for better context
- Advanced image preprocessing including:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Edge detection for better feature extraction
  - Grayscale conversion
- Custom CNN architecture optimized for emotion detection
- Data augmentation for improved model robustness

## Project Structure

- `train.py`: Training script for the emotion detection model
- `emotion_detector.py`: Core emotion detection functionality
- `realTimeDetection.py`: Real-time webcam-based emotion detection
- `requirements.txt`: Python dependencies
- `model_architecture.json`: Saved model architecture
- `facial_emotion_recognition_weights.h5`: Trained model weights
- `images/`: Training dataset directory

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ShashwatThakur0/Facial_Recognition.git
cd Facial_Recognition
```

2. Create and activate a conda environment:
```bash
conda create -n facial_recognition python=3.8
conda activate facial_recognition
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the emotion detection model:
```bash
python train.py
```

The model will be trained on the images in the `images/` directory. The best weights will be saved to `facial_emotion_recognition_weights.h5`.

### Real-time Detection

To run real-time emotion detection using your webcam:
```bash
python realTimeDetection.py
```

Press 'q' to quit the application.

## Model Architecture

The model uses a custom CNN architecture:
- 3 convolutional blocks with batch normalization
- Max pooling layers for spatial dimension reduction
- Dense layers with dropout for regularization
- Softmax output for 3-class classification

## Dataset

The system uses two types of datasets:
1. Original dataset: Real facial images labeled with emotions
2. Synthetic dataset: Generated images to augment training data

## Performance

Current model performance:
- Training accuracy: ~45%
- Validation accuracy: ~43%

## Requirements

- Python 3.8+
- OpenCV
- TensorFlow
- NumPy
- scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.