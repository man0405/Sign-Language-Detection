# Sign Language Detection using MediaPipe

This project leverages the power of MediaPipe for efficient real-time sign language detection. Initially, a CNN with LSTM model was implemented but encountered significant training times. Transitioning to MediaPipe provided a remarkable reduction in training time while maintaining high accuracy due to its advanced hand tracking and gesture recognition capabilities. MediaPipe's efficient real-time hand tracking and accurate landmark detection streamline the workflow, thereby optimizing processing time.

## Project Overview

**Data Collection**: The project collected data through video recordings referred to as sequences. Each sequence was processed to produce 30 frames, capturing the intricate details of each action. This provided a comprehensive dataset for training.

**Keypoint Extraction**: Utilizing the MediaPipe Holistic API, keypoints of the hands were identified and extracted. This step simplified the representation of hand gestures by focusing on crucial landmarks, which were then stored as numpy arrays for subsequent training.

**Model Training**: A sequential model was designed using the LSTM neural network and Computer Vision. For evaluation metrics using Adam optimizer and categorical_crossentropy loss function. The training process, benefited from MediaPipe's efficiency, resulting in significantly reduced training time. The holistic approach in extracting keypoints aided in faster and more reliable training.

![image](https://github.com/user-attachments/assets/bfe69727-05ba-42c8-a5ce-035520a8d955)

**Real-Time Testing and Visualization**: To visualize the model's performance along with the final output in form of sentences, a dynamic bar, displaying the probabilities of each predicted action in real-time detection. This allowed for intuitive understanding and validation of the model's predictions.

**Results**: The final model achieved an impressive accuracy of 99% and a very low loss, underscoring the efficacy of using MediaPipe in the context of sign language detection. MediaPipe's robust hand tracking and easy integration proved to be instrumental in achieving high performance and efficiency.

## Usage Instructions

### Setup

1. Install required dependencies:
   ```bash
   pip install torch numpy opencv-python mediapipe matplotlib
   ```

2. Organize your sign language data:
   ```bash
   python train_model.py --organize-data --split 0.8
   ```

### Training

Train the sign language recognition model:
```bash
python train_model.py --epochs 100 --batch-size 32 --learning-rate 0.001
```

Options:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--split`: Train-test split ratio if organizing data (default: 0.8)
- `--model-path`: Custom path to save the model (default: model/sign_language_model.pth)

### Real-time Detection

Run real-time sign language detection using your webcam:
```bash
python run_detection.py
```

Options:
- `--model`: Path to model file (default: model/sign_language_model.pth)
- `--camera`: Camera index to use (default: from config or 0)
- `--threshold`: Prediction confidence threshold (default: 0.7)

### Using in Notebooks

You can also use the code in Jupyter notebooks:

```python
from module.load_model_for_inference import load_model_for_inference, run_real_time_detection

# Load the model
model, actions, device = load_model_for_inference()

# Run detection
run_real_time_detection(model, actions, device, camera_idx=0, threshold=0.7)
```

## Project Structure

- **module/**
  - `data_processing.py`: Data loading and preprocessing utilities
  - `helper_functions.py`: Helper functions for file paths and configuration
  - `mediapipe_utils.py`: MediaPipe integration for keypoint extraction
  - `model_utils.py`: General model utilities
  - `sign_model_builder.py`: LSTM model architecture for sign language detection
  - `realtime_asl.py`: Real-time ASL detection with video feed
  - `load_model_for_inference.py`: Model loading and inference utilities
  - `train_lstm_model.py`: Model training utilities

- **Scripts**
  - `train_model.py`: Command-line script for model training
  - `run_detection.py`: Command-line script for running real-time detection

- **Data Directories**
  - `data_train/`: Training data organized by sign classes
  - `data_test/`: Testing data organized by sign classes

- **Notebooks**
  - `man.ipynb`: Main notebook with processing, training, and testing code
  - Other notebooks for data exploration and testing