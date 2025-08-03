import grpc
from grpc_reflection.v1alpha import reflection
import logging
from pythonjsonlogger import jsonlogger

from concurrent import futures

import proto.federated_learning_pb2 as pb2
import proto.federated_learning_pb2_grpc as pb2_grpc

# Get a logger for this module
logger = logging.getLogger(__name__)

class MLService(pb2_grpc.MLServiceServicer):
    def TrainLocalModel(self, request, context):
        logger.info("RPC call received", extra={'method': 'TrainLocalModel', 'round_id': request.round_id})
        # Dummy implementation:
        weights = pb2.ModelWeights(serialized_weights=b'', client_id='dummy', round_id=request.round_id)
        response = pb2.TrainResponse(weights=weights, loss=0.0, accuracy=0.0, confirmation=pb2.AckResponse(success=True))
        return response

    def GetModelWeights(self, request, context):
        logger.info("RPC call received", extra={'method': 'GetModelWeights'})
        # Dummy implementation:
        weights = pb2.ModelWeights(serialized_weights=b'', client_id='dummy', round_id=0)
        return weights

    def SetModelWeights(self, request, context):
        logger.info("RPC call received", extra={'method': 'SetModelWeights', 'round_id': request.global_weights.round_id})
        # Dummy implementation:
        return pb2.AckResponse(success=True, message="Weights set successfully")

    def PerformInference(self, request, context):
        logger.info("RPC call received", extra={'method': 'PerformInference'})
        # Dummy implementation:
        return pb2.ClassificationResult(label="dummy", confidence=0.0)


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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MLServiceServicer_to_server(MLService(), server)

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