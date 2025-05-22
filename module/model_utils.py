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
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Get epoch values - either from results dictionary or generate them
    epochs = results.get("epochs", range(1, len(results["train_loss"]) + 1))

    # Plot loss with grid
    axs[0].plot(epochs, results["train_loss"], 'b-',
                label="Train Loss", linewidth=2, marker='o')
    axs[0].plot(epochs, results["test_loss"], 'r-',
                label="Test Loss", linewidth=2, marker='x')
    axs[0].set_title("Loss over Epochs", fontsize=14)
    axs[0].set_xlabel("Epochs", fontsize=12)
    axs[0].set_ylabel("Loss", fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend(fontsize=12)

    # Find best test loss for annotation
    best_epoch = np.argmin(results["test_loss"])
    min_test_loss = results["test_loss"][best_epoch]
    axs[0].annotate(f'Best: {min_test_loss:.4f}',
                    xy=(epochs[best_epoch], min_test_loss),
                    xytext=(epochs[best_epoch], min_test_loss*1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=10)

    # Plot accuracy with grid
    axs[1].plot(epochs, results["train_acc"], 'g-',
                label="Train Accuracy", linewidth=2, marker='o')
    axs[1].plot(epochs, results["test_acc"], 'm-',
                label="Test Accuracy", linewidth=2, marker='x')
    axs[1].set_title("Accuracy over Epochs", fontsize=14)
    axs[1].set_xlabel("Epochs", fontsize=12)
    axs[1].set_ylabel("Accuracy", fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend(fontsize=12)

    # Find best test accuracy for annotation
    best_epoch = np.argmax(results["test_acc"])
    max_test_acc = results["test_acc"][best_epoch]
    axs[1].annotate(f'Best: {max_test_acc:.4f}',
                    xy=(epochs[best_epoch], max_test_acc),
                    xytext=(epochs[best_epoch], max_test_acc*0.9),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=10)

    # Add a title to the figure
    fig.suptitle("Training and Testing Metrics", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust for the suptitle
    plt.show()

    # Return the final metrics
    return {
        "final_train_loss": results["train_loss"][-1],
        "final_train_acc": results["train_acc"][-1],
        "final_test_loss": results["test_loss"][-1],
        "final_test_acc": results["test_acc"][-1],
        "best_test_loss": min_test_loss,
        "best_test_loss_epoch": epochs[np.argmin(results["test_loss"])],
        "best_test_acc": max_test_acc,
        "best_test_acc_epoch": epochs[np.argmax(results["test_acc"])]
    }


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
          scheduler=None,
          use_log_softmax=False,
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
        scheduler: Optional learning rate scheduler.
        use_log_softmax: Whether to apply log softmax to model outputs before loss calculation.

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

    # Store epochs for plotting
    epoch_list = list(range(1, epochs + 1))
    results["epochs"] = epoch_list

    # Training loop
    total_train_steps = len(train_dataloader) * epochs
    with tqdm(total=total_train_steps, desc="Total Progress") as pbar:
        for epoch in range(epochs):
            train_loss, train_acc = train_step(model=model,
                                               epoch=epoch,
                                               dataloader=train_dataloader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               device=device,
                                               writer=writer,
                                               use_log_softmax=use_log_softmax,
                                               pbar=pbar)

            test_loss, test_acc = test_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device,
                                            use_log_softmax=use_log_softmax)

            # Print out what's happening - only every 10 epochs or on the final epoch
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(
                    f"Epoch: {epoch+1}/{epochs} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_acc: {test_acc:.4f}"
                )

            # Update learning rate scheduler if provided
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_loss)  # For ReduceLROnPlateau
                else:
                    scheduler.step()  # For other schedulers

                # Log current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning Rate', current_lr, epoch)
                # Only print learning rate every 10 epochs or on the final epoch
                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    print(f"Current learning rate: {current_lr:.6f}")

            writer.add_scalar('Epoch/Loss/train', train_loss, epoch)
            writer.add_scalar('Epoch/Loss/test', test_loss, epoch)
            writer.add_scalar('Epoch/Accuracy/train', train_acc, epoch)
            writer.add_scalar('Epoch/Accuracy/test', test_acc, epoch)

            # Log model parameters
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(
                        f'Gradients/{name}', param.grad, epoch)

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
               use_log_softmax: bool = False,
               pbar: Optional[tqdm] = None,
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
      writer: Optional TensorBoard SummaryWriter for logging.
      use_log_softmax: Whether to apply log softmax to model outputs before loss calculation.
      pbar: Optional tqdm progress bar for updating overall progress.

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
    # Create a separate progress bar for this epoch's batches
    batch_iterator = tqdm(
        dataloader, desc=f"Epoch {epoch+1} Training", leave=False)

    # Loop through data loader data batches
    for batch_idx, (X, y) in enumerate(batch_iterator):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # Apply log softmax if specified (for use with NLLLoss)
        if use_log_softmax:
            # Add small epsilon to prevent log(0)
            y_pred = torch.log(y_pred + 1e-8)

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
        _, predicted = torch.max(
            y_pred.data if not use_log_softmax else y_pred.data, 1)
        total_train += y.size(0)
        train_acc += (predicted == y).sum().item()

        # Update batch progress bar
        batch_iterator.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{(predicted == y).sum().item() / y.size(0):.4f}'
        })

        # Update overall progress bar if provided
        if pbar is not None:
            pbar.update(1)

        # Log batch-level metrics
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Batch/Loss/train', loss.item(), global_step)

    # Adjust metrics to get average loss and accuracy per sample
    train_loss = train_loss / total_train
    train_acc = train_acc / total_train
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
              use_log_softmax: bool = False) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Args:
        model: Model to test.
        dataloader: DataLoader for testing data.
        loss_fn: Loss function.
        device: Target device ("cpu" or "cuda").
        use_log_softmax: Whether to apply log softmax to model outputs before loss calculation.

    Returns:
        Tuple of (test_loss, test_accuracy) as floats.
    """
    model.eval()  # Set to evaluation mode
    test_loss, test_acc = 0.0, 0.0
    total_samples = 0  # Track total samples processed

    # Create confusion matrix for detailed evaluation
    class_count = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():  # Disable gradient tracking
        # Wrap dataloader with tqdm for progress bar
        test_iterator = tqdm(dataloader, desc="Testing", leave=False)
        for batch_idx, (X, y) in enumerate(test_iterator):
            # Move data to device
            X, y = X.to(device), y.to(device)

            # Forward pass
            preds = model(X)

            # Track class count for confusion matrix (if first batch)
            if batch_idx == 0 and class_count == 0:
                if hasattr(model, 'num_classes'):
                    class_count = model.num_classes
                else:
                    class_count = preds.shape[1]  # Infer from output shape

            # Apply log softmax if specified (for use with NLLLoss)
            if use_log_softmax:
                # Add small epsilon to prevent log(0)
                log_preds = torch.log(preds + 1e-8)
                batch_loss = loss_fn(log_preds, y).item()
            else:
                batch_loss = loss_fn(preds, y).item()

            # Update loss (loss.item() * batch_size gives total loss for the batch)
            test_loss += batch_loss * X.size(0)

            # Update accuracy
            _, predicted = torch.max(preds, 1)
            test_acc += (predicted == y).sum().item()
            total_samples += y.size(0)  # Track total samples

            # Store predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Update progress bar
            test_iterator.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{(predicted == y).sum().item() / y.size(0):.4f}'
            })

    # Calculate average metrics
    test_loss /= total_samples
    test_acc /= total_samples

    return test_loss, test_acc
