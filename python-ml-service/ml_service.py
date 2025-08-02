import grpc
from grpc_reflection.v1alpha import reflection

from concurrent import futures

import proto.federated_learning_pb2 as pb2
import proto.federated_learning_pb2_grpc as pb2_grpc

class MLService(pb2_grpc.MLServiceServicer):
    def TrainLocalModel(self, request, context):
        print("TrainLocalModel called")
        # Dummy implementation:
        weights = pb2.ModelWeights(serialized_weights=b'', client_id='dummy', round_id=0)
        response = pb2.TrainResponse(weights=weights, loss=0.0, accuracy=0.0, confirmation=pb2.AckResponse(success=True))
        return response

    def GetModelWeights(self, request, context):
        print("GetModelWeights called")
        # Dummy implementation:
        weights = pb2.ModelWeights(serialized_weights=b'', client_id='dummy', round_id=0)
        return weights

    def SetModelWeights(self, request, context):
        print("SetModelWeights called")
        # Dummy implementation:
        return pb2.AckResponse(success=True, message="Weights set successfully")

    def PerformInference(self, request, context):
        print("PerformInference called")
        # Dummy implementation:
        return pb2.ClassificationResult(label="dummy", confidence=0.0)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MLServiceServicer_to_server(MLService(), server)

    SERVICE_NAMES = (
        pb2.DESCRIPTOR.services_by_name['MLService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("ML Service started on port 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()