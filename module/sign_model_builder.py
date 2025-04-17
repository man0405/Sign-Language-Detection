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
                 hidden_size: int = 16,
                 num_layers: int = 4,
                 num_classes: int = 2,
                 dropout_rate: float = 0.2,
                 sequence_length: int = 30) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # Implementing architecture similar to the provided Keras model
        # Using input_size for feature normalization
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.lstm1 = nn.LSTM(input_size, 16, batch_first=True, num_layers=1)

        # Using hidden size of first LSTM
        self.batch_norm2 = nn.BatchNorm1d(16)
        self.lstm2 = nn.LSTM(16, 32, batch_first=True, num_layers=1)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Using hidden size of second LSTM
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.lstm3 = nn.LSTM(32, 16, batch_first=True, num_layers=1)

        # Using hidden size of third LSTM
        self.batch_norm4 = nn.BatchNorm1d(16)
        self.lstm4 = nn.LSTM(16, 16, batch_first=True, num_layers=1)

        # Fully connected classification layers mirroring the Keras model
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * sequence_length, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1536),
            nn.ReLU(),
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

        # Applying L2 regularization similar to Keras model
        self.l2_reg_params = {
            'lstm1.weight_ih_l0': 0.001,
            'lstm1.weight_hh_l0': 0.001,
            'linear1.weight': 0.001,
            'linear2.weight': 0.001
        }

    def forward(self, x: torch.Tensor):
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Apply batch normalization (need to transpose for batch norm)
        x_bn = x.transpose(1, 2)
        x_bn = self.batch_norm1(x_bn)
        x_bn = x_bn.transpose(1, 2)

        # First LSTM layer
        out, _ = self.lstm1(x_bn)

        # Second LSTM layer with batch norm and dropout
        out_bn = out.transpose(1, 2)
        out_bn = self.batch_norm2(out_bn)
        out_bn = out_bn.transpose(1, 2)
        out, _ = self.lstm2(out_bn)
        out = self.dropout1(out)

        # Third LSTM layer with batch norm
        out_bn = out.transpose(1, 2)
        out_bn = self.batch_norm3(out_bn)
        out_bn = out_bn.transpose(1, 2)
        out, _ = self.lstm3(out_bn)

        # Fourth LSTM layer with batch norm
        out_bn = out.transpose(1, 2)
        out_bn = self.batch_norm4(out_bn)
        out_bn = out_bn.transpose(1, 2)
        out, _ = self.lstm4(out_bn)

        # Pass through the classifier
        out = self.classifier(out)
        return out
