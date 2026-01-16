import os
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def save_results(results, save_dir):
    """
    Saves the training history to a JSON file in the specified directory.
    """
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

def save_plots(history_train_loss, history_val_loss, history_val_accuracy, save_dir):
    """
    Generates and saves plots for training/validation loss and validation accuracy
    in the specified directory.
    """
    epochs = range(1, len(history_train_loss) + 1)

    # Vẽ biểu đồ Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history_train_loss, label='Training Loss', marker='o')
    plt.plot(epochs, history_val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    loss_plot_path = os.path.join(save_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close() # Đóng figure để giải phóng bộ nhớ
    print(f"Loss plot saved to {loss_plot_path}")

    # Vẽ biểu đồ Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history_val_accuracy, label='Validation Accuracy', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    acc_plot_path = os.path.join(save_dir, 'accuracy_plot.png')
    plt.savefig(acc_plot_path)
    plt.close() # Đóng figure
    print(f"Accuracy plot saved to {acc_plot_path}")
