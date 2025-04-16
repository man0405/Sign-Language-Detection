"""
Utility functions for MediaPipe-based hand and landmark detection.
"""
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    """Process an image and make predictions with the MediaPipe model.

    Args:
        image: Input BGR image
        model: MediaPipe holistic model instance

    Returns:
        image: Processed RGB->BGR image
        results: MediaPipe detection results
    """
    # Convert the BGR image to RGB and process it with MediaPipe
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    """Draw landmarks on the image.

    Args:
        image: Input image to draw on
        results: MediaPipe detection results

    Returns:
        None (modifies image in-place)
    """
    # Draw face landmarks
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10),
                               thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121),
                               thickness=1, circle_radius=1)
    )

    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(
            color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121),
                               thickness=2, circle_radius=2)
    )

    # Draw left hand landmarks
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76),
                               thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250),
                               thickness=2, circle_radius=2)
    )

    # Draw right hand landmarks
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66),
                               thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230),
                               thickness=2, circle_radius=2)
    )


def extract_keypoints(results):
    """Extract keypoints from MediaPipe results.

    Args:
        results: MediaPipe detection results

    Returns:
        np.array: Flattened array of extracted keypoints
    """
    # Extract pose landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)

    # Skipping face landmarks as requested

    # Extract left hand landmarks
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)

    # Extract right hand landmarks
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, lh, rh])


def setup_holistic_model(min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Create and configure a MediaPipe holistic model.

    Args:
        min_detection_confidence: Minimum confidence for detection
        min_tracking_confidence: Minimum confidence for tracking

    Returns:
        MediaPipe holistic model instance
    """
    return mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
