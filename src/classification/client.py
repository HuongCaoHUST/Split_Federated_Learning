import pickle
import time

from src.classification.trainers import (
    ClassificationEdgeTrainer,
    ClassificationServerTrainer,
)
from src.communication import Communication
from src.utils import create_run_dir


class ClassificationClient:
    """RabbitMQ worker for a registered classification SFL task."""

    def __init__(self, config, device, project_root, layer_id, client_id):
        self.config = config
        self.device = device
        self.project_root = project_root
        self.layer_id = layer_id
        self.client_id = client_id
        self.global_model_path = None
        self.round_index = 0
        self.dataset_descriptor = None
        self.num_batches = None
        self.run_dir = create_run_dir(project_root, layer_id, client_id)
        self.comm = Communication(config)

        time.sleep(float(config.get("startup_delay", 5)))
        self.comm.connect()
        self.client_queue_name = f"client_queue_{client_id}"
        self.comm.create_queue(self.client_queue_name)
        self.comm.send_register_message(layer_id, client_id)
        self.comm.consume_messages(self.client_queue_name, self.on_message)

    def on_message(self, ch, method, properties, body):
        try:
            payload = pickle.loads(body)
            action = payload.get("action")
            if payload.get("datasets") is not None:
                self.dataset_descriptor = payload["datasets"]
            if payload.get("nb") is not None:
                self.num_batches = int(payload["nb"])
            if payload.get("supported_cut_layers") is not None:
                self.config["cut_layer"] = list(payload["supported_cut_layers"])
            elif payload.get("cut_layer") is not None:
                self.config["cut_layer"] = [int(payload["cut_layer"])]

            if action == "start":
                self.run_trainer()
            elif action == "update_global_model":
                self.global_model_path = payload["global_model"]
                self.round_index = int(payload["round"])
                self.run_trainer()
            elif action == "stop":
                self.comm.close()
            else:
                print(f"Unknown classification action: {action}")
        except Exception as exc:
            print(f"Error processing classification message: {exc}")
            raise

    def run_trainer(self):
        if self.layer_id == 1:
            trainer = ClassificationEdgeTrainer(
                self.config,
                self.device,
                self.project_root,
                self.comm,
                self.run_dir,
                self.layer_id,
                self.client_id,
                self.dataset_descriptor,
                self.global_model_path,
                self.round_index,
            )
        elif self.layer_id == 2:
            if self.num_batches is None:
                raise ValueError("Server worker did not receive a batch allocation.")
            trainer = ClassificationServerTrainer(
                self.config,
                self.device,
                self.comm,
                self.run_dir,
                self.layer_id,
                self.client_id,
                self.num_batches,
                self.global_model_path,
                self.round_index,
            )
        else:
            raise ValueError(f"Unsupported classification layer_id={self.layer_id}.")
        trainer.run()
