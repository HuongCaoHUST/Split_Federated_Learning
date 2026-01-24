from src.communication import Communication
from src.utils import create_run_dir
import pickle

class Server:
    def __init__(self, config):
        config['rabbitmq']['host']='rabbitmq'
        self.num_client = config['clients']
        self.datasets = config['dataset']
        self.client = {}
        self.comm = Communication(config)
        self.registed = [0,0]
        self.nb_count = 0
        self.run_dir = create_run_dir('./', layer_id = 0)

    def run(self):
        print("Server class initialized.")
        self.comm.connect()
        self.comm.delete_old_queues(['intermediate_queue', 'edge_queue'])
        self.comm.create_queue('intermediate_queue')
        self.comm.create_queue('edge_queue')
        self.comm.create_queue('server_queue')
        self.comm.consume_messages('server_queue', self.on_message)

    def on_message(self, ch, method, properties, body):
        try:
            payload = pickle.loads(body)
            action = payload.get('action')

            print(f"Received action: {action}")

            if action == 'register':
                layer_id = payload.get('layer_id')
                client_id = payload.get('client_id')
                self.client[client_id] = {"layer_id": layer_id}

                if layer_id == 1:
                    self.registed[0] += 1
                else:
                    self.registed[1] += 1

                if self.registed == self.num_client:
                    self.comm.send_start_message(self.get_client_ids_by_layer(layer_id = 1))

            elif action == 'send_number_batch':
                nb = payload.get('nb_train')
                client_id = payload.get('client_id')
                self.client[client_id]["nb_train"] = nb
                self.nb_count += 1

                if self.nb_count == self.num_client[0]:
                    nb = self.get_total_nb_by_layer(layer_id = 1)
                    self.comm.send_start_message(self.get_client_ids_by_layer(layer_id = 2), nb = nb)

            elif action == 'update_model':
                model_data = payload.get('model_data')
                layer_id = payload.get('layer_id')
                epoch = payload.get('epoch')
                print(f"Updating model for Layer ID: {layer_id}")
                save_path = f"{self.run_dir}/client_layer_{layer_id}_epoch_{epoch+1}.pt"

                with open(save_path, "wb") as f:
                    f.write(model_data)

            else:
                print(f"Unknown action: {action}")

        except pickle.UnpicklingError:
            print("Error when unpack message.")
        except Exception as e:
            print(f"Error processing message: {e}")

    def get_client_ids_by_layer(self, layer_id):
        return [client_id for client_id, info in self.client.items() if info.get("layer_id") == layer_id]
    
    def get_total_nb_by_layer(self, layer_id):
        return sum(info.get("nb_train", 0) for info in self.client.values() if info.get("layer_id") == layer_id)