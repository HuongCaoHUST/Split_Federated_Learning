import os
import argparse
import uuid

from src.task import get_task_name
from src.utils import load_config_and_setup


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model.')
    parser.add_argument('--layer_id', type=int, default=-1, help='Layer ID for training.')
    parser.add_argument(
        '--config',
        default='./config.yaml',
        help='Path to a detection or classification YAML configuration.',
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    config, device = load_config_and_setup(args.config, project_root)
    task_name = get_task_name(config)
    client_id = uuid.uuid4().hex[:8]
    if task_name == 'classification':
        from src.classification.client import ClassificationClient
        from src.classification.server import ClassificationServer

        if args.layer_id in (1, 2):
            ClassificationClient(
                config, device, project_root, args.layer_id, client_id
            )
        elif args.layer_id == 0:
            ClassificationServer(config, device, project_root).run()
        else:
            parser.error('--layer_id must be 0, 1, or 2.')
    else:
        from src.client import Client
        from src.server import Server

        if args.layer_id in (1, 2):
            Client(config, device, project_root, args.layer_id, client_id)
        elif args.layer_id == 0:
            Server(config, device).run()
        else:
            parser.error('--layer_id must be 0, 1, or 2.')
