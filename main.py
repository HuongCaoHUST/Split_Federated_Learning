import os
import argparse
from src.train import train
from src.utils import load_config_and_setup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model.')
    parser.add_argument('--layer_id', type=int, default=-1, help='Layer ID for training.')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    config, device, num_classes = load_config_and_setup("./config.yaml", project_root)
    train(config, device, num_classes, project_root, args.layer_id)