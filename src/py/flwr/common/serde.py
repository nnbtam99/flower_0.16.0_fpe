# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ProtoBuf serialization and deserialization."""


from typing import Any, List, cast, Tuple

from flwr.proto.transport_pb2 import (
    ClientMessage,
    Parameters,
    Reason,
    Scalar,
    ServerMessage,
)

from . import typing

# pylint: disable=missing-function-docstring


def parameters_to_proto(parameters: typing.Parameters) -> Parameters:
    """."""
    return Parameters(tensors=parameters.tensors, tensor_type=parameters.tensor_type)


def parameters_from_proto(msg: Parameters) -> typing.Parameters:
    """."""
    tensors: List[bytes] = list(msg.tensors)
    return typing.Parameters(tensors=tensors, tensor_type=msg.tensor_type)


#  === Reconnect message ===


def reconnect_to_proto(reconnect: typing.Reconnect) -> ServerMessage.Reconnect:
    """Serialize flower.Reconnect to ProtoBuf message."""
    if reconnect.seconds is not None:
        return ServerMessage.Reconnect(seconds=reconnect.seconds)
    return ServerMessage.Reconnect()


def reconnect_from_proto(msg: ServerMessage.Reconnect) -> typing.Reconnect:
    """Deserialize flower.Reconnect from ProtoBuf message."""
    return typing.Reconnect(seconds=msg.seconds)


# === Disconnect message ===


def disconnect_to_proto(disconnect: typing.Disconnect) -> ClientMessage.Disconnect:
    """Serialize flower.Disconnect to ProtoBuf message."""
    reason_proto = Reason.UNKNOWN
    if disconnect.reason == "RECONNECT":
        reason_proto = Reason.RECONNECT
    elif disconnect.reason == "POWER_DISCONNECTED":
        reason_proto = Reason.POWER_DISCONNECTED
    elif disconnect.reason == "WIFI_UNAVAILABLE":
        reason_proto = Reason.WIFI_UNAVAILABLE
    return ClientMessage.Disconnect(reason=reason_proto)


def disconnect_from_proto(msg: ClientMessage.Disconnect) -> typing.Disconnect:
    """Deserialize flower.Disconnect from ProtoBuf message."""
    if msg.reason == Reason.RECONNECT:
        return typing.Disconnect(reason="RECONNECT")
    if msg.reason == Reason.POWER_DISCONNECTED:
        return typing.Disconnect(reason="POWER_DISCONNECTED")
    if msg.reason == Reason.WIFI_UNAVAILABLE:
        return typing.Disconnect(reason="WIFI_UNAVAILABLE")
    return typing.Disconnect(reason="UNKNOWN")


# === GetWeights messages ===


def get_parameters_to_proto() -> ServerMessage.GetParameters:
    """."""
    return ServerMessage.GetParameters()


# Not required:
# def get_weights_from_proto(msg: ServerMessage.GetWeights) -> None:


def parameters_res_to_proto(res: typing.ParametersRes) -> ClientMessage.ParametersRes:
    """."""
    parameters_proto = parameters_to_proto(res.parameters)
    return ClientMessage.ParametersRes(parameters=parameters_proto)


def parameters_res_from_proto(msg: ClientMessage.ParametersRes) -> typing.ParametersRes:
    """."""
    parameters = parameters_from_proto(msg.parameters)
    return typing.ParametersRes(parameters=parameters)


# === Fit messages ===


def fit_ins_to_proto(ins: typing.FitIns) -> ServerMessage.FitIns:
    """Serialize flower.FitIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.FitIns(parameters=parameters_proto, config=config_msg)


def fit_ins_from_proto(msg: ServerMessage.FitIns) -> typing.FitIns:
    """Deserialize flower.FitIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.FitIns(parameters=parameters, config=config)


def fit_res_to_proto(res: typing.FitRes) -> ClientMessage.FitRes:
    """Serialize flower.FitIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(res.parameters)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    # Legacy case, will be removed in a future release
    if res.num_examples_ceil is not None and res.fit_duration is not None:
        return ClientMessage.FitRes(
            parameters=parameters_proto,
            num_examples=res.num_examples,
            num_examples_ceil=res.num_examples_ceil,  # Deprecated
            fit_duration=res.fit_duration,  # Deprecated
            metrics=metrics_msg,
        )
    # Legacy case, will be removed in a future release
    if res.num_examples_ceil is not None:
        return ClientMessage.FitRes(
            parameters=parameters_proto,
            num_examples=res.num_examples,
            num_examples_ceil=res.num_examples_ceil,  # Deprecated
            metrics=metrics_msg,
        )
    # Legacy case, will be removed in a future release
    if res.fit_duration is not None:
        return ClientMessage.FitRes(
            parameters=parameters_proto,
            num_examples=res.num_examples,
            fit_duration=res.fit_duration,  # Deprecated
            metrics=metrics_msg,
        )
    # Forward-compatible case
    return ClientMessage.FitRes(
        parameters=parameters_proto,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def fit_res_from_proto(msg: ClientMessage.FitRes) -> typing.FitRes:
    """Deserialize flower.FitRes from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.FitRes(
        parameters=parameters,
        num_examples=msg.num_examples,
        num_examples_ceil=msg.num_examples_ceil,  # Deprecated
        fit_duration=msg.fit_duration,  # Deprecated
        metrics=metrics,
    )


# === Evaluate messages ===


def evaluate_ins_to_proto(ins: typing.EvaluateIns) -> ServerMessage.EvaluateIns:
    """Serialize flower.EvaluateIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.EvaluateIns(parameters=parameters_proto, config=config_msg)


def evaluate_ins_from_proto(msg: ServerMessage.EvaluateIns) -> typing.EvaluateIns:
    """Deserialize flower.EvaluateIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.EvaluateIns(parameters=parameters, config=config)


def evaluate_res_to_proto(res: typing.EvaluateRes) -> ClientMessage.EvaluateRes:
    """Serialize flower.EvaluateIns to ProtoBuf message."""
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    # Legacy case, will be removed in a future release
    if res.accuracy is not None:
        return ClientMessage.EvaluateRes(
            loss=res.loss,
            num_examples=res.num_examples,
            accuracy=res.accuracy,  # Deprecated
            metrics=metrics_msg,
        )
    # Forward-compatible case
    return ClientMessage.EvaluateRes(
        loss=res.loss,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def evaluate_res_from_proto(msg: ClientMessage.EvaluateRes) -> typing.EvaluateRes:
    """Deserialize flower.EvaluateRes from ProtoBuf message."""
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.EvaluateRes(
        loss=msg.loss,
        num_examples=msg.num_examples,
        accuracy=msg.accuracy,  # Deprecated
        metrics=metrics,
    )

# === FPE messages ===
def fpe_ins_to_proto(ins: typing.EvaluateIns) -> ServerMessage.FederatedPersonalizedEvaluateIns:
    """Serialize flower.EvaluateIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.FederatedPersonalizedEvaluateIns(parameters=parameters_proto, config=config_msg)

def fpe_ins_from_proto(msg: ServerMessage.FederatedPersonalizedEvaluateIns) -> typing.EvaluateIns:
    """Deserialize flower.EvaluateIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.EvaluateIns(parameters=parameters, config=config)    

def fpe_res_to_proto(res: Tuple[typing.EvaluateRes, typing.EvaluateRes]) -> ClientMessage.FederatedPersonalizedEvaluateRes:
    """Serialize flower.EvaluateRes to ProtoBuf message."""
    baseline, personalized = res[0], res[1]
    baseline_metrics_msg = None if baseline.metrics is None else metrics_to_proto(baseline.metrics)
    personalized_metrics_msg = None if personalized.metrics is None else metrics_to_proto(personalized.metrics)

    # Legacy case, will be removed in a future release
    if baseline.accuracy is not None:
        baseline_fpe_res_proto = ClientMessage.EvaluateRes(
            loss=baseline.loss,
            num_examples=baseline.num_examples,
            accuracy=baseline.accuracy,  # Deprecated
            metrics=baseline_metrics_msg,
        )
        personalized_fpe_res_proto = ClientMessage.EvaluateRes(
            loss=personalized.loss,
            num_examples=personalized.num_examples,
            accuracy=personalized.accuracy,  # Deprecated
            metrics=personalized_metrics_msg,
        )
        return ClientMessage.FederatedPersonalizedEvaluateRes(
            baseline=baseline_fpe_res_proto,
            personalized=personalized_fpe_res_proto
        ) 

    # Forward-compatible case

    baseline_fpe_res_proto = ClientMessage.EvaluateRes(
            loss=baseline.loss,
            num_examples=baseline.num_examples,
            metrics=baseline_metrics_msg,
        )
    personalized_fpe_res_proto = ClientMessage.EvaluateRes(
            loss=personalized.loss,
            num_examples=personalized.num_examples,
            metrics=personalized_metrics_msg,
        )
    return ClientMessage.FederatedPersonalizedEvaluateRes(
            baseline=baseline_fpe_res_proto,
            personalized=personalized_fpe_res_proto
        ) 

def fpe_res_from_proto(msg: ClientMessage.FederatedPersonalizedEvaluateRes) -> Tuple[typing.EvaluateRes, typing.EvaluateRes]:
    """Deserialize flower.EvaluateRes from ProtoBuf message."""
    baseline, personalized = msg.baseline, msg.personalized
    baseline_metrics = None if baseline.metrics is None else metrics_from_proto(baseline.metrics)
    personalized_metrics = None if personalized.metrics is None else metrics_from_proto(personalized.metrics)
    
    baseline_fpe_res = typing.EvaluateRes(
        loss=baseline.loss,
        num_examples=baseline.num_examples,
        accuracy=baseline.accuracy, # Deprecated
        metrics=baseline.metrics,
    )
    personalized_fpe_res = typing.EvaluateRes(
        loss=personalized.loss,
        num_examples=personalized.num_examples,
        accuracy=personalized.accuracy, # Deprecated
        metrics=personalized.metrics,
    )

    return baseline_fpe_res, personalized_fpe_res

# === Metrics messages ===


def metrics_to_proto(metrics: typing.Metrics) -> Any:
    """Serialize... ."""
    proto = {}
    for key in metrics:
        proto[key] = scalar_to_proto(metrics[key])
    return proto


def metrics_from_proto(proto: Any) -> typing.Metrics:
    """Deserialize... ."""
    metrics = {}
    for k in proto:
        metrics[k] = scalar_from_proto(proto[k])
    return metrics


def scalar_to_proto(scalar: typing.Scalar) -> Scalar:
    """Serialize... ."""

    if isinstance(scalar, bool):
        return Scalar(bool=scalar)

    if isinstance(scalar, bytes):
        return Scalar(bytes=scalar)

    if isinstance(scalar, float):
        return Scalar(double=scalar)

    if isinstance(scalar, int):
        return Scalar(sint64=scalar)

    if isinstance(scalar, str):
        return Scalar(string=scalar)

    raise Exception(
        f"Accepted types: {bool, bytes, float, int, str} (but not {type(scalar)})"
    )


def scalar_from_proto(scalar_msg: Scalar) -> typing.Scalar:
    """Deserialize... ."""
    scalar = getattr(scalar_msg, scalar_msg.WhichOneof("scalar"))
    return cast(typing.Scalar, scalar)
