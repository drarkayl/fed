import datetime

from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelWeights(_message.Message):
    __slots__ = ("serialized_weights", "client_id", "round_id", "timestamp")
    SERIALIZED_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    serialized_weights: bytes
    client_id: str
    round_id: int
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, serialized_weights: _Optional[bytes] = ..., client_id: _Optional[str] = ..., round_id: _Optional[int] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class TrainRequest(_message.Message):
    __slots__ = ("epochs", "batch_size", "round_id")
    EPOCHS_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    ROUND_ID_FIELD_NUMBER: _ClassVar[int]
    epochs: int
    batch_size: int
    round_id: int
    def __init__(self, epochs: _Optional[int] = ..., batch_size: _Optional[int] = ..., round_id: _Optional[int] = ...) -> None: ...

class TrainResponse(_message.Message):
    __slots__ = ("weights", "loss", "accuracy", "confirmation")
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    ACCURACY_FIELD_NUMBER: _ClassVar[int]
    CONFIRMATION_FIELD_NUMBER: _ClassVar[int]
    weights: ModelWeights
    loss: float
    accuracy: float
    confirmation: AckResponse
    def __init__(self, weights: _Optional[_Union[ModelWeights, _Mapping]] = ..., loss: _Optional[float] = ..., accuracy: _Optional[float] = ..., confirmation: _Optional[_Union[AckResponse, _Mapping]] = ...) -> None: ...

class SetGlobalWeightsRequest(_message.Message):
    __slots__ = ("global_weights",)
    GLOBAL_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    global_weights: ModelWeights
    def __init__(self, global_weights: _Optional[_Union[ModelWeights, _Mapping]] = ...) -> None: ...

class AckResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class ImageData(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class ClassificationResult(_message.Message):
    __slots__ = ("label", "confidence")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    label: str
    confidence: float
    def __init__(self, label: _Optional[str] = ..., confidence: _Optional[float] = ...) -> None: ...
