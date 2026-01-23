from src.communication import Communication
import pickle

class Server:
    def __init__(self, config):
        config['rabbitmq']['host']='rabbitmq'
        self.num_client = config['clients']
        self.datasets = config['dataset']
        self.client = {}
        self.comm = Communication(config)

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
                print(f"Received register message: {client_id}")

            elif action == 'send_number_batch':
                nb = payload.get('nb')
                client_id = payload.get('client_id')
                self.client[client_id] = {"nb": nb}

            elif action == 'update_model':
                model_data = payload.get('model_data')
                layer_id = payload.get('layer_id')
                print(f"Updating model for Layer ID: {layer_id}")

            else:
                print(f"Unknown action: {action}")

        except pickle.UnpicklingError:
            print("Error when unpack message.")
        except Exception as e:
            print(f"Error processing message: {e}")