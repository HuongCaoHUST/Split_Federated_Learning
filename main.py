import os
import argparse
from src.utils import load_config_and_setup
from src.server import Server
from src.client import Client
import uuid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model.')
    parser.add_argument('--layer_id', type=int, default=-1, help='Layer ID for training.')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    config, device = load_config_and_setup("./config.yaml", project_root)
    client_id = uuid.uuid4().hex[:8]
    if args.layer_id == 1:
        Client(config, device, project_root, args.layer_id, client_id)
    elif args.layer_id == 2:
        Client(config, device, project_root, args.layer_id, client_id)
    elif args.layer_id == 0:
        server = Server(config)
        server.run()
        