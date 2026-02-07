import mlflow
import os

class MLflowConnector:
    """
    A wrapper class to manage connection and logging to a remote MLflow server.

    This class provides explicit methods to start and end a run, which should
    be handled within a try/finally block to ensure runs are always terminated.

    Attributes:
        tracking_uri (str): The URI of the remote MLflow tracking server.
        experiment_name (str): The name of the experiment to use for logging.
    """
    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initializes the MLflowConnector and sets up the connection.

        Args:
            tracking_uri (str): The URI for the MLflow tracking server 
                                (e.g., 'http://127.0.0.1:5000').
            experiment_name (str): The name of the MLflow experiment. If it
                                   doesn't exist, it will be created.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run = None
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: str = None):
        """
        Starts an MLflow run.

        Args:
            run_name (str, optional): An optional name for the run.
        """
        self.run = mlflow.start_run(run_name=run_name)
        print(f"MLflow run '{run_name}' started (run_id: {self.run.info.run_id})")
        return self.run

    def end_run(self):
        """
        Ends the active MLflow run.
        """
        mlflow.end_run()
        print("MLflow run finished.")

    def log_param(self, key: str, value):
        """
        Logs a single parameter (e.g., learning rate, batch size).

        Args:
            key (str): The name of the parameter.
            value: The value of the parameter.
        """
        mlflow.log_param(key, value)
        print(f"Logged parameter: {{'{key}': {value}}}")

    def log_params(self, params: dict):
        """
        Logs a dictionary of parameters.

        Args:
            params (dict): A dictionary of parameters to log.
        """
        mlflow.log_params(params)
        print(f"Logged parameters: {params}")

    def log_metric(self, key: str, value: float, step: int = None):
        """
        Logs a single metric (e.g., loss, accuracy).

        Args:
            key (str): The name of the metric.
            value (float): The value of the metric.
            step (int, optional): The step or epoch for the metric.
                                  Useful for plotting over time. Defaults to None.
        """
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict, step: int = None):
        """
        Logs a dictionary of metrics at a specific step.

        Args:
            metrics (dict): A dictionary of metrics to log.
            step (int, optional): The step or epoch for the metrics. Defaults to None.
        """
        mlflow.log_metrics(metrics, step=step)
        print(f"Logged metrics at step {step}: {metrics}")

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Logs a local file or directory as an artifact.

        Args:
            local_path (str): Path to the local file or directory to log.
            artifact_path (str, optional): The directory within the MLflow run's
                                           artifact root to place the artifact.
                                           If None, it's placed in the root.
                                           Defaults to None.
        """
        mlflow.log_artifact(local_path, artifact_path)
        print(f"Logged artifact: '{local_path}'")

    def log_model(self, model, artifact_path: str, registered_model_name: str = None):
        """
        Logs a model using MLflow's model-aware logging.
        Example is for PyTorch, but can be adapted for other frameworks.
        (e.g., mlflow.sklearn.log_model, mlflow.tensorflow.log_model)

        Args:
            model: The model object (e.g., a PyTorch nn.Module).
            artifact_path (str): The path within the artifacts to save the model.
            registered_model_name (str, optional): If provided, registers the model
                                                   with this name in the Model Registry.
                                                   Defaults to None.
        """
        # Example for PyTorch. Replace with your framework if different.
        # e.g., mlflow.sklearn.log_model(model, artifact_path)
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )
        print(f"Logged model '{artifact_path}'.")


# --- Example Usage ---
if __name__ == '__main__':
    # TODO: Replace with your actual VPS IP address and port
    MLFLOW_TRACKING_URI = "http://your_vps_ip:5000"
    EXPERIMENT_NAME = "My Awesome CNN Experiment"

    # Initialize the connector
    mlflow_connector = MLflowConnector(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=EXPERIMENT_NAME
    )

    # Use a try/finally block to ensure the run is always closed
    try:
        # 1. Start the run
        mlflow_connector.start_run(run_name="Manual Run Example")
        print("Inside MLflow run context.")

        # 2. Log hyperparameters
        hyperparams = {
            "learning_rate": 0.005,
            "batch_size": 128,
            "optimizer": "SGD"
        }
        mlflow_connector.log_params(hyperparams)

        # 3. Log metrics over epochs (simulated training loop)
        print("\nSimulating training loop...")
        for epoch in range(10):
            loss = 0.8 / (epoch + 1)
            accuracy = 1.0 - loss
            mlflow_connector.log_metrics({"loss": loss, "accuracy": accuracy}, step=epoch)
        print("Finished logging metrics.\n")

        # 4. Log an artifact
        dummy_artifact_path = "dummy_artifact.txt"
        with open(dummy_artifact_path, "w") as f:
            f.write("This is a sample artifact from a manual run.")
        mlflow_connector.log_artifact(dummy_artifact_path, "configs")
        os.remove(dummy_artifact_path)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 5. End the run
        mlflow_connector.end_run()

    print("\nExample finished successfully!")
