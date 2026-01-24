import pika
import sys
import time
import pickle
import os

class Communication:
    def __init__(self, config):
        """
        Initialize Communication with the provided configuration.
        Expects config to contain key 'rabbitmq'.
        """
        self.rmq_config = config.get('rabbitmq', {})
        self.host = self.rmq_config.get('host', 'localhost')
        self.port = self.rmq_config.get('port', 5672)
        self.username = self.rmq_config.get('username', 'guest')
        self.password = self.rmq_config.get('password', 'guest')
        self.max_retries = self.rmq_config.get('max_retries', 5)
        self.retry_delay = self.rmq_config.get('retry_delay', 5)
        
        self.connection = None
        self.channel = None

    def connect(self):
        """
        Establish connection to RabbitMQ server with retry mechanism.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                print(f"Connecting to RabbitMQ at {self.host}:{self.port}...")
                credentials = pika.PlainCredentials(self.username, self.password)
                parameters = pika.ConnectionParameters(
                    host=self.host,
                    port=self.port,
                    credentials=credentials,
                    # Set a heartbeat interval to keep the connection alive
                    heartbeat=600,
                    # Set a shorter connection timeout
                    blocked_connection_timeout=300
                )
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()
                print("Successfully connected to RabbitMQ.")
                return
            except pika.exceptions.AMQPConnectionError as e:
                print(f"Error connecting to RabbitMQ: {e}")
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.retry_delay} seconds... ({retries}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                else:
                    print("Max retries reached. Could not connect to RabbitMQ.")
                    sys.exit(1)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                sys.exit(1)

    def close(self):
        """Safely close the RabbitMQ connection."""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            print("RabbitMQ connection closed.")

    def create_queue(self, queue_name):
        """
        Create (declare) a new queue.
        """
        if self.channel:
            # queue_declare is idempotent, if the queue already exists it will do nothing (unless parameters are different)
            self.channel.queue_declare(queue=queue_name)
            print(f"Queue '{queue_name}' created/declared.")
        else:
            print("Error: Channel is not initialized. Please call connect() first.")

    def delete_old_queues(self, queue_names):
        """
        Delete a list of old queues.
        Input: queue_names (list of strings) - List of queue names to delete.
        """
        if self.channel:
            for queue in queue_names:
                self.channel.queue_delete(queue=queue)
                print(f"Queue '{queue}' deleted.")
        else:
            print("Error: Channel is not initialized. Please call connect() first.")

    def publish_message(self, queue_name, message):
        """
        Publish a message to a queue.
        """
        if self.channel:
            self.channel.basic_publish(exchange='',
                                       routing_key=queue_name,
                                       body=message)
            print(f" [>>>] Sent message to '{queue_name}'")
        else:
            print("Error: Channel is not initialized. Please call connect() first.")

    def consume_message_sync(self, queue_name):
        """
        Consume a single message from a queue in a blocking manner.
        """
        if self.channel:
            while True:
                method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
                if method_frame:
                    print(f" [<<<] Received message from '{queue_name}'")
                    return body
        else:
            print("Error: Channel is not initialized. Please call connect() first.")
            return None

    def consume_messages(self, queue_name, callback):
        """
        Consume messages from a queue.
        This is a blocking call.
        """
        if self.channel:
            self.channel.basic_consume(queue=queue_name,
                                       on_message_callback=callback,
                                       auto_ack=True)
            print(' [*] Waiting for messages. To exit press CTRL+C')
            self.channel.start_consuming()
        else:
            print("Error: Channel is not initialized. Please call connect() first.")

    def send_register_message(self, layer_id = None, client_id = None):
        """
        Sends register message to centralized server.
        """
        payload = {
            'action': 'register',
            'layer_id': layer_id,
            'client_id': client_id,
        }
        self.publish_message('server_queue', pickle.dumps(payload))

    def send_start_message(self, client_ids = None, datasets = None, nb = None, nc = None, class_names = None):
        """
        Sends register message to centralized server.
        """
        for id in client_ids:
            payload = {
                'action': 'start'
            }
            if datasets is not None: payload['datasets'] = datasets
            if nb and nc is not None:
                payload['nb'] = nb
                payload['nc'] = nc
                payload['class_names'] = class_names
                
            self.publish_message(f'client_queue_{id}', pickle.dumps(payload))

    def send_training_metadata(self, queue_name, client_id, nb_train = None, nb_val = None):
        """
        Sends training metadata (number of training and validation batches) to a queue.
        """
        payload = {
            'action': 'send_number_batch',
            'client_id': client_id,
            'nb_train': nb_train,
            'nb_val': nb_val
        }
        self.publish_message(queue_name, pickle.dumps(payload))

    def publish_model(self, queue_name, model_path, layer_id = None, epoch = None):

        try:
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                return

            with open(model_path, 'rb') as f:
                model_data = f.read()

                payload = {
                    'action': 'update_model',
                    'model_data': model_data,
                    'layer_id': layer_id,
                    'epoch': epoch
                }
            
            self.publish_message(queue_name, pickle.dumps(payload))
            print(f"Successfully published model from {model_path} to '{queue_name}'")
        except Exception as e:
            print(f"An error occurred while publishing the model: {e}")