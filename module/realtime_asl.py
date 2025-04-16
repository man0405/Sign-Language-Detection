"""
Functionality for real-time ASL detection and speech synthesis.
"""
import cv2
import time
import pyttsx3
import torch
import numpy as np
import os

from module.mediapipe_utils import mediapipe_detection, draw_landmarks, extract_keypoints, setup_holistic_model

class ASLDetector:
    """Real-time American Sign Language detector with speech synthesis."""
    
    def __init__(self, model_path="model/sign_language_model.pth", 
                 threshold=0.7, sequence_length=30, device="cpu"):
        """Initialize the ASL detector.
        
        Args:
            model_path: Path to the trained model
            threshold: Confidence threshold for predictions
            sequence_length: Number of frames to use for prediction
            device: Device to run the model on
        """
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.device = device
        
        # Load model if it exists
        self.model = None
        self.actions = None
        self.load_model(model_path)
        
        # Initialize TTS engine
        self.engine = pyttsx3.init()
        
        # Other variables
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.last_spoken = ""
        self.last_spoken_time = 0

    def load_model(self, model_path):
        """Load a trained PyTorch model.
        
        Args:
            model_path: Path to the trained model
        """
        try:
            # Try loading with weights_only=False for PyTorch 2.6+ compatibility
            try:
                from module.sign_model_builder import LSTM_Sign_Model
                # Register the model class as safe
                torch.serialization.add_safe_globals([LSTM_Sign_Model])
                self.model = torch.load(model_path, map_location=self.device)
            except Exception as first_error:
                # Fallback to the older method if the above fails
                try:
                    self.model = torch.load(model_path, map_location=self.device, weights_only=False)
                    print("Model loaded with weights_only=False for backward compatibility")
                except Exception:
                    raise first_error
            
            self.model.eval()
            print(f"Loaded model from {model_path}")
            
            # Try to get actions from data directory
            data_path = os.path.join('data')
            if os.path.exists(data_path):
                self.actions = [action for action in os.listdir(data_path) 
                              if not action.startswith('.')]
                print(f"Detected actions: {self.actions}")
            else:
                print("Warning: Data directory not found. Actions list is empty.")
                self.actions = []
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.actions = []

    def speak(self, text):
        """Convert text to speech.
        
        Args:
            text: Text to convert to speech
        """
        self.engine.say(text)
        self.engine.runAndWait()

    def detect_in_realtime(self, camera_idx=0):
        """Run real-time ASL detection.
        
        Args:
            camera_idx: Camera index to use
        """
        if self.model is None:
            print("Error: No model loaded. Cannot perform detection.")
            return
        
        # Initialize webcam
        cap = cv2.VideoCapture(camera_idx)
        
        # Set up MediaPipe model
        with setup_holistic_model() as holistic:
            while cap.isOpened():
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                    
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                
                # Draw landmarks
                draw_landmarks(image, results)
                
                # Extract keypoints for prediction
                keypoints = extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-self.sequence_length:]
                
                # Make prediction when we have enough frames
                if len(self.sequence) == self.sequence_length:
                    # Prepare input for the model
                    input_data = torch.tensor(np.expand_dims(self.sequence, axis=0), 
                                             dtype=torch.float32).to(self.device)
                    
                    # Get prediction
                    with torch.no_grad():
                        outputs = self.model(input_data)
                        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                        pred_class = torch.argmax(probs).item()
                        pred_prob = probs[pred_class].item()
                    
                    # Get the top 3 predictions
                    top3_probs, top3_indices = torch.topk(probs, min(3, len(self.actions)))
                    top3_actions = [(self.actions[i.item()], p.item()) 
                                   for i, p in zip(top3_indices, top3_probs)]
                        
                    # Check if prediction is confident enough
                    if pred_prob > self.threshold:
                        if len(self.sentence) > 0:
                            # Only add to sentence if it's a new prediction
                            if self.actions[pred_class] != self.sentence[-1]:
                                self.sentence.append(self.actions[pred_class])
                        else:
                            self.sentence.append(self.actions[pred_class])
                    
                    # Keep last 5 recognized signs
                    if len(self.sentence) > 5:
                        self.sentence = self.sentence[-5:]
                    
                    # Speak the detected sign if it's new and enough time has passed
                    current_time = time.time()
                    if self.sentence and self.sentence[-1] != self.last_spoken and current_time - self.last_spoken_time > 2:
                        self.speak(self.sentence[-1])
                        self.last_spoken = self.sentence[-1]
                        self.last_spoken_time = current_time
                
                # Display prediction and probabilities
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(self.sentence), (3, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show the prediction probabilities
                if len(self.sequence) == self.sequence_length and 'top3_actions' in locals():
                    for i, (action, prob) in enumerate(top3_actions):
                        cv2.putText(image, f"{action}: {prob:.2f}", (500, 70 + i*40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show the frame
                cv2.imshow('ASL Detection', image)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()

def run_asl_detection():
    """Main function to run ASL detection."""
    # Try to load the model
    detector = ASLDetector()
    
    # Run real-time detection
    detector.detect_in_realtime()