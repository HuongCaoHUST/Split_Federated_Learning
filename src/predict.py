
import torch
import argparse
import yaml
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import sys
import os
import time

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Alexnet import AlexNet
from model.Mobilenet import MobileNet
from model.VGG16 import VGG16
from src.utils import _load_class_names_from_file, count_parameters # Import from utils

# Supported image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

def predict_image(model, device, preprocess, class_names, input_channels, image_path):
    """
    Runs prediction on a single image using a trained model.
    """
    try:
        image = Image.open(image_path).convert('RGB') if input_channels == 3 else Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Input image not found at '{image_path}'")
        return None, None, None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None

    image_tensor = preprocess(image)

    # Determine model input channels safely
    model_in_channels = 3 # Default assumption
    if hasattr(model, 'features') and len(model.features) > 0 and isinstance(model.features[0], torch.nn.Conv2d):
        model_in_channels = model.features[0].in_channels
    elif hasattr(model, 'conv1') and isinstance(model.conv1, torch.nn.Conv2d):
        model_in_channels = model.conv1.in_channels
    elif hasattr(model, 'conv1_1') and isinstance(model.conv1_1, torch.nn.Conv2d):
        model_in_channels = model.conv1_1.in_channels

    # If model expects 3 channels but input is 1, repeat the channel
    if model_in_channels == 3 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Perform Inference
    start_time = time.time()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    end_time = time.time()
    
    predicted_class = class_names[predicted_idx.item()]
    inference_time = end_time - start_time

    return predicted_class, confidence.item(), inference_time

def predict_source(opt, model, device, preprocess, class_names, input_channels, model_name):
    """
    Handles prediction for either a single image or a directory of images.
    """
    image_files = []
    if os.path.isfile(opt.source):
        image_files.append(opt.source)
    elif os.path.isdir(opt.source):
        for root, _, files in os.walk(opt.source):
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    image_files.append(os.path.join(root, file))
    else:
        print(f"Error: Source '{opt.source}' is neither a file nor a directory.")
        sys.exit(1)

    if not image_files:
        print(f"No supported image files found in '{opt.source}'")
        sys.exit(1)

    all_predictions = []
    total_inference_time = 0
    total_images = len(image_files)

    print(f"\n--- Predicting for {'single image' if total_images == 1 else 'images'} in: {opt.source} ---")
    
    for i, img_path in enumerate(image_files):
        predicted_class, confidence, inference_time = predict_image(
            model, device, preprocess, class_names, input_channels, img_path
        )
        if predicted_class:
            all_predictions.append({
                "image": os.path.basename(img_path),
                "prediction": predicted_class,
                "confidence": confidence,
                "inference_time": inference_time,
                "index": i + 1,
                "total": total_images
            })
            total_inference_time += inference_time

    print(f"\nModel Name: {model_name}")
    print(f"Model Parameters: {count_parameters(model):,}")
    print(f"Number of Images: {total_images}")
    print(f"Input Size: 224x224")
    print(f"Total Inference Speed: {total_inference_time:.4f} seconds")
    
    if total_images > 1:
        print("\nIndividual Predictions:")
        for pred in all_predictions:
            print(f"  [{pred['index']}/{pred['total']}] Image: {pred['image']}, Prediction: {pred['prediction']}, Confidence: {pred['confidence']:.4f}, Time: {pred['inference_time']:.4f}s")
    elif total_images == 1:
        # For a single image, print directly without "Individual Predictions" header
        pred = all_predictions[0]
        print(f"Prediction: {pred['prediction']}")
        print(f"Confidence: {pred['confidence']:.4f}")
        print(f"Inference Speed: {pred['inference_time']:.4f} seconds")
    
    print("-----------------------------------------------------")


def predict(opt):
    """
    Runs prediction on a single image or a directory of images using a trained model.
    """
    # --- 1. Load Configuration and Class Names ---
    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config.get('model', {}).get('name', 'AlexNet')
    dataset_name = config.get('dataset', {}).get('name', 'MNIST')
    
    # Define class names based on the dataset or load from file
    if dataset_name.upper() == 'MNIST':
        class_names = [str(i) for i in range(10)]
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        input_channels = 1
    elif dataset_name.upper() == 'CIFAR10':
        class_names_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'class_names.py')
        class_names = _load_class_names_from_file(class_names_file_path)
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
    elif model_name == 'VGG16':
        model = VGG16(num_classes=num_classes)
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
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # --- 4. Handle Source (Single Image or Directory) ---
    predict_source(opt, model, device, preprocess, class_names, input_channels, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run prediction on an image or a directory of images.")
    parser.add_argument('--source', type=str, required=True, help='Path to the input image or a directory of images.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained model weights (.pth).')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    
    args = parser.parse_args()
    predict(args)
