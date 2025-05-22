import torch.nn as nn
import torch.nn.functional as F


class SignLanguageLSTM(nn.Module):
    """LSTM model for sign language recognition as used in the notebook.

    Args:
        input_size: An integer indicating number of features in input sequences.
        hidden_sizes: A list of integers for the sizes of the LSTM layers.
        num_classes: An integer indicating number of output classes.
    """

    def __init__(self, input_size, hidden_sizes, num_classes):
        super(SignLanguageLSTM, self).__init__()
        # Define the three LSTM layers
        self.lstm1 = nn.LSTM(
            input_size, hidden_sizes[0], num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(
            hidden_sizes[0], hidden_sizes[1], num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(
            hidden_sizes[1], hidden_sizes[2], num_layers=1, batch_first=True)
        # Define the dense layers
        self.fc1 = nn.Linear(hidden_sizes[2], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Pass through the first LSTM and apply ReLU
        out, _ = self.lstm1(x)
        out = F.relu(out)
        # Pass through the second LSTM and apply ReLU
        out, _ = self.lstm2(out)
        out = F.relu(out)
        # Pass through the third LSTM, take the last time step, and apply ReLU
        out, _ = self.lstm3(out)
        out = out[:, -1, :]  # Shape: (batch_size, hidden_sizes[2])
        out = F.relu(out)
        # Pass through the dense layers with ReLU
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # Output layer with softmax
        out = self.fc3(out)
        out = F.softmax(out, dim=1)
        return out
