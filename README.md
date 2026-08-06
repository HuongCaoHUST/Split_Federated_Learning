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

## Canonical-gradient baseline: uniform cut 5

`config_canonical_cut5.yaml` is a deliberately narrow correctness baseline for
one edge and one server worker.  It uses a **single full YOLO11 model and a
single optimizer on the server**.  The edge only executes layers `0..5`, then
sends the two boundary activations; after receiving their gradients, it sends
the edge-prefix gradients back to the server.  The server installs those
gradients into its canonical full model and calls `optimizer.step()` exactly
once per batch.

This is therefore not model averaging and must not be used with more than one
edge yet.  It is intended to verify that splitting at layer 5 changes only the
place of computation, not the mathematical update.

First run the local numerical check (it uses a synthetic detection batch and
does not require RabbitMQ):

```bash
python verify_canonical_cut5.py
```

Expected output includes `Canonical cut-5 verification PASSED` and zero
difference for the loss, gradients, and post-step state.  To run the actual
one-edge/one-server experiment, start RabbitMQ and then use three terminals:

```bash
python main.py --layer_id 0 --config config_canonical_cut5.yaml
python main.py --layer_id 1 --config config_canonical_cut5.yaml
python main.py --layer_id 2 --config config_canonical_cut5.yaml
```

The coordinator validates the final checkpoint as a normal full `YOLO11_Full`
model.  For a fair accuracy comparison, hold pretrained checkpoint, data split,
batch order, optimizer, learning rate, augmentation, and number of optimizer
steps fixed; compare this experiment with an unsplit full-model run.  The
implementation currently rejects any cut other than `[5]` or any client shape
other than `[1, 1]` rather than silently applying an invalid aggregation rule.

Alternatively, Docker Compose can start all three workers and RabbitMQ with the
canonical configuration:

```bash
docker compose -f docker-compose.yml -f docker-compose.canonical-cut5.yml up --build
```

Use `Ctrl-C` to stop the foreground run.  The command uses
`config_canonical_cut5.yaml`; the normal `docker-compose.yml` remains the
dynamic-cut configuration.

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


The trained models and validation results are saved in a `runs` directory created automatically.

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
