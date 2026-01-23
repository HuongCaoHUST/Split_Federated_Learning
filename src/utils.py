import os
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys # Added for sys.exit
import yaml
import torch
import psutil
import gc

def update_results_csv(epoch, train_loss, val_loss=None, val_accuracy=None, save_dir = './results'):
    """
    Appends the latest epoch results to a CSV file.
    Creates the file and writes the header on the first call.
    """
    results_path = os.path.join(save_dir, 'results.csv')
    file_exists = os.path.isfile(results_path)

    with open(results_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy'])
        
        writer.writerow([epoch, train_loss, val_loss, val_accuracy])
    # No print statement here to avoid cluttering the epoch log

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

def _load_class_names_from_file(file_path):
    """
    Loads class names from a specified file.
    Assumes the file contains a line like: CLASSES = ('class1', 'class2', ...)
    """
    class_names = None
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Use a dictionary to capture the exec'd variables
        exec_globals = {}
        exec(content, exec_globals)
        
        if 'CLASSES' in exec_globals and isinstance(exec_globals['CLASSES'], tuple):
            class_names = exec_globals['CLASSES']
        else:
            raise ValueError(f"Could not find 'CLASSES' tuple in {file_path}")
    except FileNotFoundError:
        print(f"Error: Class names file not found at '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading class names from {file_path}: {e}")
        sys.exit(1)
    return class_names

def count_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_run_dir(project_root, layer_id=None):
    """
    Creates a new directory for the current run to save results.
    """
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    run_idx = 1
    while True:
        if layer_id is not None:
            run_name = f"run_{layer_id}_{run_idx}"
        else:
            run_name = f"run_{run_idx}"
        run_dir = os.path.join(results_dir, run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            break
        run_idx += 1
    print(f"Created run directory: {run_dir}")
    return run_dir

def load_config_and_setup(config_path, project_root):
    """
    Tải cấu hình, thiết lập device và trả về các thông số.
    """
    # Tải cấu hình từ file config.yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    return config, device

def get_memory(unit='fraction'):
        """
        unit: fraction, gb, bytes
        """
        cgroup_v1_current = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        cgroup_v1_max = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

        try:
            limit = None
            usage = None

            if os.path.exists(cgroup_v1_current):
                with open(cgroup_v1_current, "r") as f:
                    usage = int(f.read().strip())
            
            if os.path.exists(cgroup_v1_max):
                with open(cgroup_v1_max, "r") as f:
                    limit = int(f.read().strip())

            if usage is not None:
                if limit is None or limit > 10**15: 
                    limit = psutil.virtual_memory().total
                
                if unit == 'fraction':
                    return usage / limit
                elif unit == 'gb':
                    return usage / (1024**3)
                else:
                    return usage 

        except Exception as e:
            print(f"Warning reading cgroup: {e}") 
            pass

        mem = psutil.virtual_memory()
        if unit == 'fraction':
            return mem.percent / 100.0
        elif unit == 'gb':
            return mem.used / (1024**3)
        return mem.used

def clear_memory(device, threshold: float = 0.85):
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if get_memory(unit='fraction') <= threshold:
                return

        gc.collect()
        
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()