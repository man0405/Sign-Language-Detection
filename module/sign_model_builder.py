"""
Contains PyTorch model code to instantiate an LSTM model for sign language recognition.
"""
import torch
from torch import nn 

class LSTM_Sign_Model(nn.Module):
    """Creates the LSTM Sign Language Recognition Model architecture.

    This model uses stacked LSTM layers followed by fully connected layers
    to classify sequences of hand landmarks into sign language gestures.

    Args:
        input_size: An integer indicating number of features in input sequences.
        hidden_size: An integer indicating number of hidden units in LSTM layers.
        num_layers: An integer indicating number of stacked LSTM layers.
        num_classes: An integer indicating number of output classes.
        dropout_rate: A float indicating dropout rate (default: 0.2).
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 num_classes: int,
                 dropout_rate: float = 0.2) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm_stack = nn.Sequential(
            nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True),
            nn.LSTM(hidden_size, hidden_size*2, num_layers=1, batch_first=True),
            nn.LSTM(hidden_size*2, hidden_size, num_layers=1, batch_first=True)
        )
        
        # Fully connected classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Need to handle the LSTM layers separately due to the hidden state returns
        out, _ = self.lstm_stack[0](x)
        out, _ = self.lstm_stack[1](out)
        out, _ = self.lstm_stack[2](out)
        
        # We only need the output from the last time step
        out = out[:, -1, :]
        
        # Pass through the classifier
        out = self.classifier(out)
        return out