from src.communication import Communication

class Server:
    def __init__(self, config):
        config['rabbitmq']['host']='rabbitmq'
        self.comm = Communication(config)

    def run(self):
        print("Server class initialized.")
        self.comm.connect()
        self.comm.delete_old_queues(['intermediate_queue', 'edge_queue'])
        self.comm.create_queue('intermediate_queue')
        self.comm.create_queue('edge_queue')