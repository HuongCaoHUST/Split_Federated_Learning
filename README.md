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

## Configurable classification SFL

The classification path is selected by `task: classification` and runs beside
the existing YOLO detection path. Dataset and split-model implementations are
selected through registries, so trainers and the coordinator do not depend on
AlexNet or MNIST directly. MNIST and CIFAR-10 are currently registered as
datasets; AlexNet, ResNet18, and MobileNetV2 are registered dynamic split
models.

Use the provided configuration with Docker Compose:

```bash
SFL_CONFIG=config_classification.yaml \
docker compose up --build --scale torch_1=4
```

The default `SFL_CONFIG` remains `config.yaml`, so the existing detection
command continues to use YOLO. To run the classification processes directly,
start RabbitMQ and then launch one coordinator, four edge workers, and one
dynamic-server worker:

```bash
python main.py --layer_id 0 --config config_classification.yaml
python main.py --layer_id 1 --config config_classification.yaml  # four terminals
python main.py --layer_id 2 --config config_classification.yaml
```

For processes running on the host instead of in Compose, set
`rabbitmq.host: localhost` in the classification config.

Choose the dataset and model in one config:

```yaml
model:
  name: ResNet18  # AlexNet, ResNet18, or MobileNetV2
  num_classes: 10

dataset:
  name: CIFAR10  # or MNIST
  split_dir: auto
  input_size: 224
  channels: 3
```

With `split_dir: auto`, changing `dataset.name` is enough to select the correct
download, transforms, validation set, split script, and shard directory.
Valid split points are 0 through 7 for AlexNet, 0 through 5 for ResNet18, and
0 through 9 for MobileNetV2. Results are written to
`classification_results.csv` with `best.pt`,
`last.pt`, and `classification_split_graph.txt`.

### Split MNIST among clients

Create four non-IID index shards using a label-based Dirichlet distribution:

```bash
python scripts/split_mnist_dirichlet.py \
  --mode dirichlet --alpha 0.5 --num-clients 4 --seed 42
```

For a balanced IID split:

```bash
python scripts/split_mnist_dirichlet.py \
  --mode iid --num-clients 4 --seed 42
```

### Split CIFAR-10 among clients

The CIFAR-10 splitter has the same IID/Dirichlet options and keeps the official
10,000-image test set as shared validation data:

```bash
python scripts/split_cifar10_dirichlet.py \
  --mode dirichlet --alpha 0.5 --num-clients 4 --seed 42
```

For an IID split, replace `--mode dirichlet` with `--mode iid`.

Both splitters produce one `client_N_indices.pt` shard per client, class count
and ratio CSV files, a complete index manifest, JSON metadata, and
`class_distribution_heatmap.png`. You can use an explicit shard path:

```yaml
dataset:
  name: CIFAR10
  split_dir: data/cifar10_splits/cifar10_dirichlet_alpha_0p5
```

At startup, the coordinator validates that the shards are disjoint, reads each
client's index list and class statistics, and sends only that descriptor to the
matching edge client through RabbitMQ. The client then constructs its local
`Subset(dataset, indices)`; raw images are not copied through the broker.
When `dataset.auto_create_split: true`, a completely missing or empty
`split_dir` is generated automatically using `split_mode`, `dirichlet_alpha`,
`split_seed`, and `subset_fraction` from the configuration. An existing but
incomplete directory is never overwritten automatically.

Use `--overwrite` to intentionally replace a split generated earlier with the
same output path. The official MNIST or CIFAR-10 test set is not partitioned.

To add another split model, register its full, edge, and dynamic-server classes
in `src/classification/models.py`. The implementation must expose global
stages through `layers`, declare `LAYER_NAMES` and `SUPPORTED_CUT_LAYERS`, and
use local `layers` indices in edge/server checkpoints for generic aggregation.

### Centralized classification baseline

The centralized classification baseline trains the complete model on the full
CIFAR10 training set and evaluates on the official CIFAR10 test set. It does
not use RabbitMQ, client shards, or split layers.

Run the provided AlexNet baseline from the repository root:

```bash
python train_classification_centralized.py \
  --config config_classification_centralized.yaml
```

For a quick CPU smoke run:

```bash
python train_classification_centralized.py \
  --config config_classification_centralized.yaml \
  --epochs 1 --device cpu
```

The first run downloads CIFAR10 into `data/`. Results are written to the
configured output directory, including `best.pt`, `last.pt`, and
`classification_centralized_results.csv`. The trainer supports `AlexNet`,
`ResNet18`, and `MobileNetV2` through the `model.name` setting, although the
provided centralized config is intended for the AlexNet CIFAR10 baseline.

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
