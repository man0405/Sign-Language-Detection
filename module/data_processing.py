"""
Data processing utilities for ASL detection system.
"""
import os
import cv2
import time
import torch
import numpy as np
import shutil
import random
from torch.utils.data import Dataset, DataLoader

from module.mediapipe_utils import mediapipe_detection, extract_keypoints, draw_landmarks
from module.helper_functions import get_username, get_data_path


class SignLanguageDataset(Dataset):
    """PyTorch Dataset for Sign Language keypoints."""

    def __init__(self, data_dir, target_frames=30, input_size=225):
        self.data_dir = data_dir
        self.target_frames = target_frames  # Target number of frames per sequence
        # Number of features per frame (99 pose + 63 left hand + 63 right hand)
        self.input_size = input_size
        self.classes = [cls for cls in os.listdir(
            data_dir) if not cls.startswith('.')]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Find all sequences
        self.sequences = []
        self.labels = []

        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for seq_id in os.listdir(cls_dir):
                if not seq_id.startswith('.') and os.path.isdir(os.path.join(cls_dir, seq_id)):
                    seq_path = os.path.join(cls_dir, seq_id)
                    # Check if directory has at least one .npy file before adding
                    npy_files = [f for f in os.listdir(
                        seq_path) if f.endswith('.npy')]
                    if len(npy_files) > 0:
                        self.sequences.append(seq_path)
                        self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.sequences[idx]
        label = self.labels[idx]

        # Load all frames in sequence
        frames = []
        frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.npy')],
                             key=lambda x: int(x.split('.')[0]))

        for frame_file in frame_files:
            frame_path = os.path.join(seq_path, frame_file)
            try:
                frame = np.load(frame_path)
                # Keep the full frame data (pose + hands)
                frames.append(frame)
            except Exception as e:
                print(f"Error loading {frame_path}: {e}")
                continue

        # Ensure we have the right number of frames with proper handling
        if len(frames) == 0:
            # No valid frames, return zeros with the right shape
            return torch.zeros((self.target_frames, self.input_size), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        # Check that all frames have the same shape
        shapes = [frame.shape[0] for frame in frames]
        if len(set(shapes)) > 1:
            # If frames have different shapes, use only those with the most common shape
            common_shape = max(set(shapes), key=shapes.count)
            frames = [frame for frame in frames if frame.shape[0] == common_shape]

        # Handle sequences with different lengths
        if len(frames) < self.target_frames:
            # Pad short sequences by repeating the last frame
            if len(frames) > 0:
                last_frame = frames[-1]
                padding = [last_frame] * (self.target_frames - len(frames))
                frames.extend(padding)
            else:
                # No frames, generate empty ones
                return torch.zeros((self.target_frames, self.input_size), dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        elif len(frames) > self.target_frames:
            # Truncate long sequences
            frames = frames[:self.target_frames]

        # Stack frames into a single array
        sequence = np.stack(frames)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class SignLanguageFolderDataset(Dataset):
    """PyTorch Dataset for Sign Language folder structure."""

    def __init__(self, data_dir, target_frames=30, input_size=225):
        self.data_dir = data_dir
        self.target_frames = target_frames  # Target number of frames per sequence
        # Number of features per frame (99 pose + 63 left hand + 63 right hand)
        self.input_size = input_size
        self.classes = [cls for cls in os.listdir(
            data_dir) if not cls.startswith('.')]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Find all sequences
        self.sequences = []
        self.labels = []

        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for seq_id in os.listdir(cls_dir):
                if not seq_id.startswith('.') and os.path.isdir(os.path.join(cls_dir, seq_id)):
                    seq_path = os.path.join(cls_dir, seq_id)
                    # Check if directory has at least one .npy file before adding
                    npy_files = [f for f in os.listdir(
                        seq_path) if f.endswith('.npy')]
                    if len(npy_files) > 0:
                        self.sequences.append(seq_path)
                        self.labels.append(self.class_to_idx[cls])

        print(f"Loaded {len(self.sequences)} valid sequences from {data_dir}")
        if len(self.sequences) == 0:
            print("WARNING: No valid sequences found!")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.sequences[idx]
        label = self.labels[idx]

        # Load sequence frames
        frame_paths = [os.path.join(seq_path, f) for f in sorted(os.listdir(seq_path))
                       if not f.startswith('.') and f.endswith('.npy')]

        if len(frame_paths) == 0:
            # Handle empty sequences by creating a zero tensor
            print(f"Warning: Empty sequence found at {seq_path}")
            return torch.zeros((self.target_frames, self.input_size), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        # Load frames safely
        frames = []
        for frame_path in frame_paths:
            try:
                frame_data = np.load(frame_path)
                if frame_data.size > 0:  # Check if array is not empty
                    # Keep the entire frame data (pose + hands)
                    frames.append(frame_data)
            except Exception as e:
                print(f"Error loading {frame_path}: {e}")
                continue

        # If no valid frames were loaded, return zeros
        if len(frames) == 0:
            print(f"Warning: No valid frames in {seq_path}")
            return torch.zeros((self.target_frames, self.input_size), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        # Handle sequences with different lengths
        if len(frames) < self.target_frames:
            # Pad short sequences by repeating the last frame
            last_frame = frames[-1]
            padding = [last_frame] * (self.target_frames - len(frames))
            frames.extend(padding)
        elif len(frames) > self.target_frames:
            # Truncate long sequences
            frames = frames[:self.target_frames]

        # Stack frames
        try:
            sequence = np.stack(frames)
        except ValueError as e:
            print(f"Error stacking frames in {seq_path}: {e}")
            # Get first frame shape or use default
            if frames:
                frame_shape = frames[0].shape[0]
            else:
                frame_shape = self.input_size  # Use the expected input size

            # Create an empty sequence with the right shape
            sequence = np.zeros((self.target_frames, frame_shape))

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def collect_sign_data(sign_name, holistic_model, num_sequences=30, sequence_length=30, camera_idx=0):
    """Collect sign language data for a new sign.

    Args:
        sign_name: Name of the sign to collect data for (e.g., 'hello', 'thanks')
        holistic_model: MediaPipe holistic model instance
        num_sequences: Number of sequences to collect
        sequence_length: Number of frames per sequence
        camera_idx: Camera index to use
    """
    # Get username from config file
    username = get_username()

    # Create directory structure if it doesn't exist
    data_dir = os.path.join(get_data_path(), sign_name)
    os.makedirs(data_dir, exist_ok=True)

    # Find the highest existing sequence number to continue from there
    existing_sequences = [int(seq.split('_')[-1]) if '_' in seq else int(seq)
                          for seq in os.listdir(data_dir)
                          if (seq.isdigit() or (username and seq.startswith(f"{username}_")))
                          and os.path.isdir(os.path.join(data_dir, seq))]
    start_sequence = 0
    if existing_sequences:
        start_sequence = max(existing_sequences) + 1
        print(
            f"Found existing sequences. Starting from sequence {start_sequence}")

    # Loop through sequences
    for sequence in range(start_sequence, start_sequence + num_sequences):
        # Create directory for this sequence with username prefix if available
        if username:
            sequence_dir = os.path.join(data_dir, f"{username}_{sequence}")
            display_sequence = f"{username}_{sequence}"
        else:
            sequence_dir = os.path.join(data_dir, str(sequence))
            display_sequence = str(sequence)

        os.makedirs(sequence_dir, exist_ok=True)

        # Start webcam capture
        cap = cv2.VideoCapture(camera_idx)

        # Collect frames for one sequence
        for frame_num in range(sequence_length):
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic_model)

            # Draw landmarks
            draw_landmarks(image, results)

            # Display collection progress with username if available
            if username:
                display_text = f'Collecting frames for {sign_name} - User: {username} - Sequence {sequence}/{start_sequence + num_sequences - 1} - Frame {frame_num+1}/{sequence_length}'
            else:
                display_text = f'Collecting frames for {sign_name} - Sequence {sequence}/{start_sequence + num_sequences - 1} - Frame {frame_num+1}/{sequence_length}'

            cv2.putText(image, display_text, (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Wait logic
            if frame_num == 0:
                # Wait for 5 seconds before starting each sequence
                cv2.putText(image, "Starting collection in 5 seconds...", (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(5000)
            else:
                # Small delay between frames
                cv2.waitKey(100)

            # Extract keypoints and save
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(sequence_dir, f'{frame_num}.npy')
            np.save(npy_path, keypoints)

            # Break loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # Release webcam after each sequence
        cap.release()
        cv2.destroyAllWindows()

        # Wait between sequences
        if sequence < start_sequence + num_sequences - 1:
            print(
                f"Sequence {display_sequence} complete. Prepare for next sequence...")
            time.sleep(3)

    print(f"Data collection for sign '{sign_name}' complete!")
    user_info = f" for user '{username}'" if username else ""
    print(
        f"Collected {num_sequences} sequences{user_info} with {sequence_length} frames each.")
    print(f"Data saved in {os.path.abspath(data_dir)}")
    print(
        f"Total sequences for '{sign_name}': {start_sequence + num_sequences}")


def organize_data_for_testing(train_split=0.7):
    """Organize data by moving some sequences to a test folder.

    Args:
        train_split: Proportion of data to use for training (0.0 to 1.0)
    """
    # Create train and test directories using helper functions
    train_dir = os.path.join('data_train')
    test_dir = os.path.join('data_test')

    # Delete existing directories if they exist
    if os.path.exists(train_dir):
        print(f"Removing existing directory: {train_dir}")
        shutil.rmtree(train_dir)

    if os.path.exists(test_dir):
        print(f"Removing existing directory: {test_dir}")
        shutil.rmtree(test_dir)

    # Create fresh directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of signs from main data directory
    data_dir = os.path.join(get_data_path())
    signs = [sign for sign in os.listdir(data_dir) if not sign.startswith('.')]

    for sign in signs:
        # Create sign directories in train and test
        os.makedirs(os.path.join(train_dir, sign), exist_ok=True)
        os.makedirs(os.path.join(test_dir, sign), exist_ok=True)

        # Get all sequence directories
        sign_dir = os.path.join(data_dir, sign)
        sequences = [seq for seq in os.listdir(sign_dir)
                     if not seq.startswith('.') and os.path.isdir(os.path.join(sign_dir, seq))]

        # Shuffle and split sequences
        random.shuffle(sequences)
        split_idx = int(len(sequences) * train_split)
        train_sequences = sequences[:split_idx]
        test_sequences = sequences[split_idx:]

        # Copy sequences to train directory
        for seq in train_sequences:
            src = os.path.join(sign_dir, seq)
            dst = os.path.join(train_dir, sign, seq)
            if not os.path.exists(dst):
                shutil.copytree(src, dst)

        # Copy sequences to test directory
        for seq in test_sequences:
            src = os.path.join(sign_dir, seq)
            dst = os.path.join(test_dir, sign, seq)
            if not os.path.exists(dst):
                shutil.copytree(src, dst)

    print(f"Data organized into {train_dir} and {test_dir} directories")
    print(
        f"Training data: {train_split*100:.0f}%, Testing data: {(1-train_split)*100:.0f}%")


def create_dataloaders(train_dir, test_dir, batch_size=32, num_workers=0):
    """
    Create dataloaders for training and testing datasets.
    This function creates PyTorch DataLoader objects for both training and testing datasets.
    It uses the SignLanguageDataset class to load the data from the specified directories.
    The datasets are assumed to be organized in a folder structure where each subfolder
    corresponds to a different class of sign language gestures.

    Args:
        train_dir: Directory containing training data.
        test_dir: Directory containing testing data.
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for DataLoader    

    Returns:
        train_dataloader, test_dataloader, class_names
    """

    train_dataset = SignLanguageDataset(train_dir)
    test_dataset = SignLanguageDataset(test_dir)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader, train_dataset.classes


def create_separate_dataloaders(train_dir, test_dir, batch_size=32, num_workers=0):
    """Create separate training and testing dataloaders.

    Args:
               train_dir: Directory containing training data
        test_dir: Directory containing testing data
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for DataLoader

    Returns:
        train_dataloader, test_dataloader, class_names
    """

    # Create datasets using folder dataset
    train_dataset = SignLanguageFolderDataset(train_dir)
    test_dataset = SignLanguageFolderDataset(test_dir)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader, train_dataset.classes
