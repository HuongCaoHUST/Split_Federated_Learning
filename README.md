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
    -   `pretrained_path`: Path to a pretrained model to start from.
-   `cut_layer`: One split index per edge client. A single value is broadcast to the group; otherwise the list length must equal `clients[0]`. Scaled replicas receive values in registration order. The dynamic server starts after the minimum configured cut and routes each intermediate payload according to its client cut (for example `[5, 10]`).
    During aggregation, every global layer is weighted by the number of batches that actually traversed its edge or server copy. This includes overlapping layers that run on the server for earlier cuts and on the edge for later cuts.
    Intermediate payloads carry their global epoch, so faster clients cannot mix next-epoch updates into the current aggregation round.
    Coordinator checkpoints include both client ID and epoch, preventing replicas at the same layer from overwriting one another.
-   `dataset`: Path to the dataset configuration YAML file(s).
-   `rabbitmq`: Connection details for the RabbitMQ server.

## Centralized YOLO11 baseline

The centralized trainer uses `model.YOLO11_Full` directly with the complete
dataset. After each epoch, it logs the following metrics to MLflow:

- Training and validation `box_loss`, `cls_loss`, and `dfl_loss`.
- Precision, Recall, mAP50, and mAP50-95.

### 1. Configuration

Edit the parameters in `config_centralized.yaml` before starting. At minimum,
set the correct path to the full dataset YAML:

```yaml
dataset:
  yaml: datasets/full.yaml

training:
  epochs: 100
  batch_size: 8
  device: auto

mlflow:
  tracking_uri: http://smart-hvac.io.vn:5005/
  experiment_name: Centralized_YOLO11
```

The dataset YAML must use the YOLO format and contain `train`, `val`, `nc`, and
`names`. Relative paths in `config_centralized.yaml` are resolved from the
directory containing the configuration file.

### 2. Build the Docker image

From the repository root, run:

```bash
docker compose --profile centralized build centralized
```

Rebuild the image only when the Dockerfile or dependencies change.

### 3. Run on CPU

Set `training.device: cpu` in `config_centralized.yaml`, then run:

```bash
docker compose --profile centralized run --rm centralized
```

### 4. Run on an NVIDIA GPU

The host must have an NVIDIA driver and the NVIDIA Container Toolkit installed.
Set `training.device: cuda:0` or `training.device: auto`, then run:

```bash
docker compose --profile centralized-gpu run --rm centralized-gpu
```

### 5. Use a different configuration file

You can keep multiple experiment configurations and select one at runtime
without editing `docker-compose.yml`:

```bash
CENTRALIZED_CONFIG=config_centralized_experiment_2.yaml \
docker compose --profile centralized-gpu run --rm centralized-gpu
```

### 6. Results

When `output.directory` is `null`, results are saved automatically under:

```text
runs/centralized/<run-name>/
├── best.pt
├── last.pt
└── metrics.csv
```

`metrics.csv`, `best.pt`, `last.pt`, and the dataset YAML are also uploaded as
MLflow artifacts after training finishes.

The trainer can also run directly without Docker:

```bash
python train_centralized.py --config config_centralized.yaml
```

CLI arguments take precedence over YAML values when a quick override is needed:

```bash
python train_centralized.py \
  --config config_centralized.yaml \
  --epochs 10 \
  --device cpu
```

## Split the Living Room Dataset

Use `scripts/split_livingroom_dirichlet.py` to create four YOLO client datasets. The
script keeps each train image with its label, copies the complete `valid` split to
every client, writes one YAML file per client, and creates a class-distribution
heatmap.

```bash
# Non-IID: lower alpha creates stronger label skew
python scripts/split_livingroom_dirichlet.py \
  --mode dirichlet --alpha 0.5 --num-clients 4 --seed 42

# IID: balanced random split
python scripts/split_livingroom_dirichlet.py \
  --mode iid --alpha 0.5 --num-clients 4 --seed 42 \
  --output datasets/subdataset/livingroom_2_iid
```

The generated YAML files and client data are stored below
`datasets/subdataset/`. Use the corresponding `client_1.yaml` through
`client_4.yaml` files when launching the four clients. The YAML files use
relative paths, so the complete output directory can be zipped and extracted
elsewhere without editing the configurations.
