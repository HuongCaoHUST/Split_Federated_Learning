import os
from src.train import train
from src.utils import load_config_and_setup

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    config, device, num_classes = load_config_and_setup("./config.yaml", project_root)
    train(config, device, num_classes, project_root)