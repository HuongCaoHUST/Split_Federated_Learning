from src.communication import Communication
from src.train import TrainerEdge, TrainerServer
import time
import pickle

class Client:
    def __init__(self, config, device, project_root, layer_id, client_id):
        self.comm = Communication(config)

        self.config = config
        self.device = device
        self.project_root = project_root
        self.layer_id = layer_id
        self.client_id = client_id

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
            self.datasets = payload.get('datasets')
            self.nb = payload.get('nb')
            self.nc = payload.get('nc')
            self.class_names = payload.get('class_names')

            print(f"Received action: {action}")

            if action == 'start':
                self.run()
            else:
                print(f"Unknown action: {action}")

        except pickle.UnpicklingError:
            print("Error when unpack message.")
        except Exception as e:
            print(f"Error processing message: {e}")

    def run(self):
        print("Client class initialized.")
        if self.layer_id == 1:
            time.sleep(10)
            trainer = TrainerEdge(self.config, self.device, self.project_root, self.comm, self.layer_id, self.client_id, self.datasets)
            trainer.run()
        elif self.layer_id == 2:
            time.sleep(10)
            trainer = TrainerServer(self.config, self.device, self.project_root, self.comm, self.layer_id, self.client_id, self.nb, self.nc, self.class_names)
            trainer.run()
        else:
            print(f"Error layer id: {self.layer_id}")