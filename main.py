import os
import argparse
from src.train import train
from src.utils import load_config_and_setup
import uuid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model.')
    parser.add_argument('--layer_id', type=int, default=-1, help='Layer ID for training.')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    config, device = load_config_and_setup("./config.yaml", project_root)
    client_id = uuid.uuid4().hex[:8]
    train(config, device, project_root, args.layer_id, client_id)