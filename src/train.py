import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm

from model.Alexnet import AlexNet
from model.Mobilenet import MobileNet
from src.utils import update_results_csv, save_plots, count_parameters, create_run_dir
from src.dataset_loader import DatasetLoader

def train(config, device, num_classes, project_root):
    # Create a new run directory
    run_dir = create_run_dir(project_root)
    
    # Create Dataloader
    dataset_loader = DatasetLoader(config, project_root)
    train_loader, validation_loader = dataset_loader.get_loaders()

    # Set yperparameters
    NUM_EPOCHS = config['training']['num_epochs']
    LEARNING_RATE = config['training']['learning_rate']
    MODEL_NAME = config['model']['name']
    MODEL_SAVE_PATH = config['model']['save_path']
    SAVE_MODEL = config['model'].get('save_model', True) # Get new config option

    # Initialize model
    print(f"Initializing model: {MODEL_NAME}...")
    model_map = {
        'AlexNet': AlexNet,
        'MobileNet': MobileNet
    }

    if MODEL_NAME not in model_map:
        print(f"Error: Model '{MODEL_NAME}' not recognized. Supported models: {list(model_map.keys())}")
        sys.exit(1)
    model = model_map[MODEL_NAME](num_classes=num_classes).to(device)

    print(f"Model Parameters: {count_parameters(model):,}") # Log model parameters here
    print("Model initialized.")

    # Init Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
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
        
        # Log to CSV
        update_results_csv(epoch + 1, avg_train_loss, avg_val_loss, val_accuracy, run_dir)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    print("Finished Training.")

    # Save model
    if SAVE_MODEL: # Conditional saving
        save_path = os.path.join(run_dir, MODEL_SAVE_PATH)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    else:
        print("Model saving skipped as per configuration.")

    print("Saving plots...")
    save_plots(history_train_loss, history_val_loss, history_val_accuracy, run_dir)
    print("Plots saved.")
