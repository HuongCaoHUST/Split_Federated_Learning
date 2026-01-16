import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import sys
import os
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import SubsetRandomSampler

# Thêm đường dẫn thư mục gốc của dự án vào sys.path
# để có thể import các module từ thư mục 'model'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Alexnet import AlexNet
from model.Mobilenet import MobileNet
from src.utils import save_results, save_plots

def create_run_dir(project_root):
    """
    Creates a new directory for the current run to save results.
    """
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Find the next available run directory
    run_idx = 1
    while os.path.exists(os.path.join(results_dir, f'run_{run_idx}')):
        run_idx += 1
    
    run_dir = os.path.join(results_dir, f'run_{run_idx}')
    os.makedirs(run_dir)
    
    print(f"Created run directory: {run_dir}")
    return run_dir

def load_config_and_setup(project_root):
    """
    Tải cấu hình, thiết lập device và trả về các thông số.
    """
    # Tải cấu hình từ file config.yaml
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    return config, device

def prepare_data(config, project_root):
    """
    Chuẩn bị và chia dữ liệu, trả về DataLoaders.
    """
    print("Preparing data...")
    data_path = os.path.join(project_root, 'data')
    d_config = config['dataset']
    validation_split = d_config['validation_split']
    subset_fraction = d_config.get('subset_fraction', 1.0) # Dùng get để tương thích ngược
    batch_size = config['training']['batch_size']

    transform = transforms.Compose(
        [transforms.Resize((227, 227)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                                 download=True, transform=transform)

    # Tạo samplers để chia train/validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))
    
    np.random.seed(42) # Để đảm bảo chia giống nhau mỗi lần chạy
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Lấy một phần nhỏ của dataset nếu được chỉ định
    if subset_fraction < 1.0:
        train_subset_size = int(len(train_indices) * subset_fraction)
        val_subset_size = int(len(val_indices) * subset_fraction)
        train_indices = train_indices[:train_subset_size]
        val_indices = val_indices[:val_subset_size]
        print(f"Using a subset of the data: {subset_fraction*100:.1f}%")

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, num_workers=2)
    print("Data prepared.")
    print(f"{len(train_indices)} training images, {len(val_indices)} validation images.")
    
    return train_loader, validation_loader

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create a new run directory
    run_dir = create_run_dir(project_root)

    # =============================================================================
    # 1. & 2. Tải cấu hình và Chuẩn bị dữ liệu
    # =============================================================================
    config, device = load_config_and_setup(project_root)
    train_loader, validation_loader = prepare_data(config, project_root)

    # Lấy các hyperparameters từ config
    NUM_EPOCHS = config['training']['num_epochs']
    LEARNING_RATE = config['training']['learning_rate']
    MODEL_NAME = config['model']['name']
    MODEL_SAVE_PATH = config['model']['save_path']

    # =============================================================================
    # 3. Khởi tạo Model, Loss và Optimizer
    # =============================================================================
    print(f"Initializing model: {MODEL_NAME}...")
    if MODEL_NAME == 'AlexNet':
        model = AlexNet(num_classes=10).to(device)
    elif MODEL_NAME == 'MobileNet':
        model = MobileNet(num_classes=10).to(device)
    else:
        print(f"Error: Model '{MODEL_NAME}' not recognized. Exiting.")
        sys.exit()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model initialized.")

    # =============================================================================
    # 4. Vòng lặp Training và Validation
    # =============================================================================
    print("Starting Training...")
    history_train_loss = []
    history_val_loss = []
    history_val_accuracy = []

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for images, labels in train_progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            val_progress_bar = tqdm(validation_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
            for images, labels in val_progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(validation_loader)
        val_accuracy = 100 * correct / total
        history_val_loss.append(avg_val_loss)
        history_val_accuracy.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    print("Finished Training.")

    # =============================================================================
    # 5. Lưu Model
    # =============================================================================
    save_path = os.path.join(run_dir, MODEL_SAVE_PATH)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # =============================================================================
    # 6. Lưu kết quả và vẽ biểu đồ
    # =============================================================================
    results = {
        'train_loss': history_train_loss,
        'val_loss': history_val_loss,
        'val_accuracy': history_val_accuracy
    }
    save_results(results, run_dir)
    save_plots(history_train_loss, history_val_loss, history_val_accuracy, run_dir)
