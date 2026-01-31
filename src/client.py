from src.communication import Communication
from src.train import TrainerEdge, TrainerServer
import time
import pickle
from src.utils import create_run_dir

class Client:
    def __init__(self, config, device, project_root, layer_id, client_id):
        self.comm = Communication(config)

        self.config = config
        self.device = device
        self.project_root = project_root
        self.layer_id = layer_id
        self.client_id = client_id
        self.global_model_path = None
        self.run_dir = create_run_dir(project_root, layer_id, client_id)
        self.round = 0

        time.sleep(5)
        self.comm.connect()
        self.client_queue_name = f'client_queue_{client_id}'
        self.comm.create_queue(self.client_queue_name)
        self.comm.send_register_message(layer_id, client_id)
        self.comm.consume_messages(self.client_queue_name, self.on_message)

    def on_message(self, ch, method, properties, body):
        try:
            payload = pickle.loads(body)
            action = payload.get('action')
            if payload.get('datasets') is not None: self.datasets = payload.get('datasets')
            if payload.get('nb') is not None: self.nb = payload.get('nb')
            if payload.get('nc') is not None: self.nc = payload.get('nc')
            if payload.get('class_names') is not None: self.class_names = payload.get('class_names')

            print(f"Received action: {action}")
            if action == 'start':
                self.run()
            elif action == 'update_global_model':
                self.global_model_path = payload.get('global_model')
                self.round = payload.get('round')
                self.run()
            elif action == 'stop':
                self.comm.close()
            else:
                print(f"Unknown action: {action}")

        except pickle.UnpicklingError:
            print("Error when unpack message.")
        except Exception as e:
            print(f"Error processing message: {e}")

    def run(self):
        print("Client class initialized.")
        if self.layer_id == 1:
            time.sleep(1)
            trainer = TrainerEdge(self.config, self.device, self.project_root, self.comm, self.run_dir, self.layer_id, self.client_id, self.datasets, self.global_model_path, self.round)
            trainer.run()
        elif self.layer_id == 2:
            time.sleep(1)
            trainer = TrainerServer(self.config, self.device, self.project_root, self.comm, self.run_dir, self.layer_id, self.client_id, self.nb, self.nc, self.class_names, self.global_model_path, self.round)
            trainer.run()
        else:
            print(f"Error layer id: {self.layer_id}")