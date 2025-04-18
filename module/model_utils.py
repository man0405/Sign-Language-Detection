"""
Utilities for model training, evaluation, and visualization.
"""
import os
from typing import Tuple, Optional
import torch

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter

from module.helper_functions import get_model_path, get_runs_path


def set_seeds(seed=42):
    """Set seeds for reproducibility.

    Args:
        seed: Seed value for random number generators
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_loss_curves(results):
    """Plot training and validation loss curves and accuracy curves.

    Args:
        results: Dictionary containing loss and accuracy values
    """
    # Create figure with 2x1 subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Get epoch values
    epochs = range(len(results["train_loss"]))

    # Plot loss
    axs[0].plot(epochs, results["train_loss"], label="Train Loss")
    axs[0].plot(epochs, results["test_loss"], label="Test Loss")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot accuracy
    axs[1].plot(epochs, results["train_acc"], label="Train Accuracy")
    axs[1].plot(epochs, results["test_acc"], label="Test Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def save_model(model, path=None):
    """Save a PyTorch model to a file.

    Args:
        model: PyTorch model to save
        path: Path to save the model to. If None, will use default path from get_model_path().
    """
    if path is None:
        # Use default name in the configured model directory
        model_dir = get_model_path()
        path = os.path.join(model_dir, 'sign_language_model.pth')
    else:
        # If path is provided but doesn't include directory, use the configured model directory
        if os.path.dirname(path) == '':
            model_dir = get_model_path()
            path = os.path.join(model_dir, path)
        # If path is provided with a directory that matches the pattern 'model/',
        # replace it with the configured model directory
        elif os.path.dirname(path).replace('\\', '/').startswith('model/'):
            relative_path = os.path.basename(path)
            model_dir = get_model_path()
            path = os.path.join(model_dir, relative_path)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save model
    torch.save(model, path)
    print(f"Model saved as '{path}'")


def load_model(path=None, device='cpu'):
    """Load a PyTorch model from a file.

    Args:
        path: Path to the saved model. If None, will use default path from get_model_path().
        device: Device to load the model to

    Returns:
        Loaded PyTorch model
    """
    if path is None:
        # Use default name in the configured model directory
        model_dir = get_model_path()
        path = os.path.join(model_dir, 'sign_language_model.pth')
    else:
        # If path is provided but doesn't include directory, use the configured model directory
        if os.path.dirname(path) == '':
            model_dir = get_model_path()
            path = os.path.join(model_dir, path)
        # If path is provided with a directory that matches the pattern 'model/',
        # replace it with the configured model directory
        elif os.path.dirname(path).replace('\\', '/').startswith('model/'):
            relative_path = os.path.basename(path)
            model_dir = get_model_path()
            path = os.path.join(model_dir, relative_path)

    if os.path.exists(path):
        model = torch.load(path, map_location=device)
        print(f"Loaded model from '{path}'")
        return model
    else:
        print(f"No model found at '{path}'")
        return None


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs=50,
          device="cpu",
          writer: Optional[SummaryWriter] = None,
          ):
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        writer: Optional TensorBoard SummaryWriter for logging.

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]} 
        For example if training for epochs=2: 
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]} 
    """
    if writer is None:
        import datetime
        runs_dir = get_runs_path()
        log_dir = os.path.join(
            runs_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to {log_dir}")

    # Move model to device
    model.to(device)

    # Initialize metrics tracking
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Training loop
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           epoch=epoch,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                           writer=writer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        writer.add_scalar('Epoch/Loss/train', train_loss, epoch)
        writer.add_scalar('Epoch/Loss/test', test_loss, epoch)
        writer.add_scalar('Epoch/Accuracy/train', train_acc, epoch)
        writer.add_scalar('Epoch/Accuracy/test', test_acc, epoch)

        # Log model parameters
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


def train_step(model: torch.nn.Module,
               epoch: int,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               writer: Optional[SummaryWriter] = None,
               ) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
      model: A PyTorch model to be trained.
      dataloader: A DataLoader instance for the model to be trained on.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A tuple of training loss and training accuracy metrics.
      In the form (train_loss, train_accuracy). For example:

      (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    total_train = 0
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)

        # Apply L2 regularization if the model has l2_reg_params defined
        if hasattr(model, 'l2_reg_params'):
            l2_reg = 0.0
            for name, param in model.named_parameters():
                if name in model.l2_reg_params:
                    l2_reg += model.l2_reg_params[name] * \
                        torch.norm(param) ** 2
            loss += l2_reg

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Accumulate loss and sample count
        train_loss += loss.item() * X.size(0)

        # Calculate and accumulate accuracy
        _, predicted = torch.max(y_pred.data, 1)
        total_train += y.size(0)
        train_acc += (predicted == y).sum().item()

        # Log batch-level metrics
        if writer is not None:
            global_step = epoch * len(dataloader) + batch
            writer.add_scalar('Batch/Loss/train', loss.item(), global_step)

    # Adjust metrics to get average loss and accuracy per sample
    train_loss = train_loss / total_train
    train_acc = train_acc / total_train
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Args:
        model: Model to test.
        dataloader: DataLoader for testing data.
        loss_fn: Loss function.
        device: Target device ("cpu" or "cuda").

    Returns:
        Tuple of (test_loss, test_accuracy) as floats.
    """
    model.eval()  # Set to evaluation mode
    test_loss, test_acc = 0.0, 0.0
    total_samples = 0  # Track total samples processed

    with torch.inference_mode():  # Disable gradient tracking
        # Wrap dataloader with tqdm for progress bar
        for batch_idx, (X, y) in enumerate(tqdm(dataloader, desc="Testing", leave=False)):
            # Move data to device
            X, y = X.to(device), y.to(device)

            # Forward pass
            preds = model(X)

            # Update loss (loss.item() * batch_size gives total loss for the batch)
            batch_loss = loss_fn(preds, y).item()
            test_loss += batch_loss * X.size(0)

            # Update accuracy
            _, predicted = torch.max(preds, 1)
            test_acc += (predicted == y).sum().item()
            total_samples += y.size(0)  # Track total samples

    # Calculate average metrics
    test_loss /= total_samples
    test_acc /= total_samples

    return test_loss, test_acc
