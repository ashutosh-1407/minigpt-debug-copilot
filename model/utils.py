import torch
import matplotlib.pyplot as plt


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def plot_train_val_loss_curve(train_losses: list[float], val_losses: list[float]) -> None:
    steps = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, "b", label="Training loss")
    plt.plot(steps, val_losses, "r", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
