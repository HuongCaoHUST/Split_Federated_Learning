
import torch
import argparse
import yaml
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import sys
import os

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Alexnet import AlexNet
from model.Mobilenet import MobileNet

def predict(opt):
    """
    Runs prediction on a single image using a trained model.
    """
    # --- 1. Load Configuration and Class Names ---
    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config.get('model', {}).get('name', 'AlexNet')
    dataset_name = config.get('dataset', {}).get('name', 'MNIST')
    
    # Define class names based on the dataset
    if dataset_name.upper() == 'MNIST':
        class_names = [str(i) for i in range(10)]
        # MNIST images are grayscale, but AlexNet expects 3 channels.
        # We will need to convert the input image to 3 channels.
        # Normalization values for MNIST
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        input_channels = 1
    elif dataset_name.upper() == 'CIFAR10':
        class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # Normalization values for CIFAR-10
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        input_channels = 3
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Please use 'MNIST' or 'CIFAR10'.")

    num_classes = len(class_names)

    # --- 2. Load Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'AlexNet':
        model = AlexNet(num_classes=num_classes)
    elif model_name == 'MobileNet':
        model = MobileNet(num_classes=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    try:
        model.load_state_dict(torch.load(opt.weights, map_location=device))
    except FileNotFoundError:
        print(f"Error: Weights file not found at '{opt.weights}'")
        sys.exit(1)
        
    model.to(device)
    model.eval()

    # --- 3. Image Preprocessing ---
    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # Load and process image
    try:
        image = Image.open(opt.source).convert('RGB') if input_channels == 3 else Image.open(opt.source).convert('L')
    except FileNotFoundError:
        print(f"Error: Input image not found at '{opt.source}'")
        sys.exit(1)

    image_tensor = preprocess(image)

    # If model expects 3 channels but input is 1, repeat the channel
    if model.features[0].in_channels == 3 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # --- 4. Perform Inference ---
    import time
    start_time = time.time()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    end_time = time.time()
    
    predicted_class = class_names[predicted_idx.item()]
    inference_time = end_time - start_time

    # --- 5. Display Results ---
    print(f"Model Name: {model_name}")
    print(f"Number of Images: 1")
    print(f"Input Size: 224x224")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence.item():.4f}")
    print(f"Inference Speed: {inference_time:.4f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run prediction on an image.")
    parser.add_argument('--source', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained model weights (.pth).')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    
    args = parser.parse_args()
    predict(args)
