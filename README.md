# Split Federated Learning for YOLOv11 Object Detection

This project implements a Split Federated Learning (SFL) system to train a YOLOv11 object detection model. The core idea is to split the neural network model into multiple parts, distributing them between edge clients and a central server. This allows for training on decentralized data without sharing the raw data itself, enhancing privacy and efficiency.

## Architecture

The system follows a client-server architecture orchestrated by a message broker.

- **Server**: The central coordinator that manages the federated learning rounds. Its responsibilities include:
  - Registering and managing clients.
  - Aggregating (merging) model parts received from all clients.
  - Performing validation on the reconstructed global model.
  - Logging metrics to MLflow.
  - Sending the updated global model back to the clients for the next round.

- **Clients (Edge Devices)**: Each client holds a partition of the YOLOv11 model. There are two types of clients corresponding to the two main splits of the model. They perform the following tasks:
  - Train their model partition on local data.
  - Send the updated model part back to the server.
  - Receive the new global model from the server.

- **RabbitMQ**: Acts as the message bus for all asynchronous communication between the server and the clients.

The training process is as follows:
1. The server and all clients connect to the RabbitMQ service.
2. Clients register themselves with the server.
3. Once all clients are registered, the server initiates the first training round.
4. Each client trains its part of the model and sends the updated weights to the server.
5. The server waits for all client updates, merges them to form a new global model, and validates it.
6. The server sends the new model back to the clients, starting the next round. This repeats for a configured number of rounds.

## Project Structure

```
.
├── config.yaml              # Main configuration file (model split, training params)
├── docker-compose.yml       # Docker Compose for services like RabbitMQ
├── main.py                  # Main entry point to start server or clients
├── requirements.txt         # Python dependencies
├── run_train.sh             # Example shell script to execute training
├── model/                   # Model definitions (e.g., YOLO11n_custom.py)
└── src/                     # Source code
    ├── server.py            # Server logic
    ├── client.py            # Client logic
    ├── train.py             # Training loops for edge and server parts
    ├── communication.py     # RabbitMQ communication handler
    ├── dataset.py           # Dataset handling
    ├── predict.py           # Script for running predictions
    └── ...
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HuongCaoHUSTSplit_Federated_Learning.git
    cd Split_Federated_Learning
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Start RabbitMQ Service:**
    This project is configured to connect to a RabbitMQ instance. You can easily start one using the provided Docker Compose file.
    ```bash
    docker-compose up -d rabbitmq
    ```

## How to Run

The training process requires one server instance and multiple client instances running concurrently in separate terminals. The number of clients should match the configuration in `config.yaml`.

1.  **Terminal 1: Start the Server**
    ```bash
    python main.py --layer_id 0
    ```

2.  **Terminal 2: Start Client Layer 1 1**
    This client handles the first part of the model.
    ```bash
    python main.py --layer_id 1
    ```

3.  **Terminal 3: Start Client Layer 2**
    This client handles the second part of the model.
    ```bash
    python main.py --layer_id 2
    ```
    
    *Note: If you configure more clients in `config.yaml`, you will need to open more terminals to run them.*

## Configuration

The main behavior of the system is controlled by `config.yaml`.

-   `clients`: A list defining the number of clients of each type. For example, `[1, 1]` means one client for part 1 and one for part 2.
-   `training`: Parameters for the training process like `num_epochs`, `num_rounds`, `batch_size`, and `learning_rate`.
-   `model`: Defines the model architecture.
    -   `cut_layer`: The index of the layer where the model is split. This is a critical parameter for Split Learning.
    -   `pretrained_path`: Path to a pretrained model to start from.
-   `dataset`: Path to the dataset configuration YAML file(s).
-   `rabbitmq`: Connection details for the RabbitMQ server.


The trained models and validation results are saved in a `runs` directory created automatically.
