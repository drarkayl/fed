import grpc
from grpc_reflection.v1alpha import reflection
import logging
from pythonjsonlogger import jsonlogger
import threading

from concurrent import futures

import proto.federated_learning_pb2 as pb2
import proto.federated_learning_pb2_grpc as pb2_grpc

import ml_model.ml_model_cnn as ml_model


# Get a logger for this module
logger = logging.getLogger(__name__)

class MLService(pb2_grpc.MLServiceServicer):
    def __init__(self, model, train_dataset, val_dataset):
        """
        Initializes the MLService with a single model instance and datasets.

        Args:
            model: The machine learning model instance.
            train_dataset: Dataset for training.
            val_dataset: Dataset for validation.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # Reverse mapping from class index to class name for inference results.
        self.idx_to_class = {v: k for k, v in self.train_dataset.class_to_idx.items()}
        # lock to ensure thread-safe access to the model.
        self.model_lock = threading.Lock()
        self.client_id = "python-ml-service-node-1"  # find a better way to manage client IDs in a real implementation
        self.round_id = 0  # rethink how to manage round IDs in a real implementation

    def TrainLocalModel(self, request, context):
        logger.info("RPC call received", extra={'method': 'TrainLocalModel', 'round_id': request.round_id})

        with self.model_lock:
            # The train_model function modifies the model in-place.

            logger.info("Setting up data loaders...", extra={'batch_size': request.batch_size})
            train_loader, val_loader = ml_model.create_dataloaders(self.train_dataset, self.val_dataset, request.batch_size)

            _, epoch_val_acc, epoch_val_loss = ml_model.train_model(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=request.epochs,
                device="cpu"
            )

            # Serialize the newly trained model weights.
            model_bytes = ml_model.serialize_model(self.model)

        response_weights = pb2.ModelWeights(
            serialized_weights=model_bytes,
            client_id=self.client_id,
            round_id=request.round_id
        )

        response = pb2.TrainResponse(
            weights=response_weights,
            loss=epoch_val_loss,
            accuracy=epoch_val_acc.item(),
            confirmation=pb2.AckResponse(success=True, message="Training completed successfully.")
        )

        return response

    def GetModelWeights(self, request, context):
        logger.info("RPC call received", extra={'method': 'GetModelWeights'})
        with self.model_lock:
            model_bytes = ml_model.serialize_model(self.model)

        weights = pb2.ModelWeights(serialized_weights=model_bytes, client_id=self.client_id, round_id=self.round_id)
        return weights

    def SetModelWeights(self, request, context):
        logger.info("RPC call received", extra={'method': 'SetModelWeights', 'round_id': request.global_weights.round_id})
        with self.model_lock:
            ml_model.deserialize_model(
                model=self.model,
                model_bytes=request.global_weights.serialized_weights,
                device="cpu"
            )
        return pb2.AckResponse(success=True, message="Global weights set successfully")

    def PerformInference(self, request, context):
        logger.info("RPC call received", extra={'method': 'PerformInference'})
        # A lock is used to prevent reading the model while it's being updated.
        with self.model_lock:
            predicted_class_idx, confidence = ml_model.perform_inference(
                model=self.model,
                image_data=request.data,
                device="cpu"
            )

        label = self.idx_to_class.get(predicted_class_idx, "Unknown")

        return pb2.ClassificationResult(label=label, confidence=confidence)


def setup_logging():
    """Configures structured (JSON) logging."""
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Create a file handler to write logs to a file
    log_handler = logging.FileHandler("python-ml-service.log")

    # Create a JSON formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )

    # Set the formatter for the handler and add it to the root logger
    log_handler.setFormatter(formatter)
    if not root_logger.handlers:
        root_logger.addHandler(log_handler)


def serve():
    setup_logging()

    logger.info("Setting up model...")
    model = ml_model.create_model(ml_model.NUM_CLASSES)

    logger.info("Creating datasets...")
    train_dataset, val_dataset = ml_model.create_datasets(ml_model.TRAIN_DIR, ml_model.VAL_DIR, ml_model.TARGET_IMG_SIZE)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Pass the single instance of the model and datasets to the servicer
    pb2_grpc.add_MLServiceServicer_to_server(
        MLService(model=model, train_dataset=train_dataset, val_dataset=val_dataset),
        server
    )

    SERVICE_NAMES = (
        pb2.DESCRIPTOR.services_by_name['MLService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("ML Service started", extra={'port': 50051})
    server.wait_for_termination()


if __name__ == '__main__':
    serve()