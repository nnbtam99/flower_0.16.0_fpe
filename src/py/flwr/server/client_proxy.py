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
"""Flower client (abstract base class)."""


from abc import ABC, abstractmethod

from flwr.common import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,
    Reconnect,
)

from typing import Tuple

class ClientProxy(ABC):
    """Abstract base class for Flower client proxies."""

    def __init__(self, cid: str):
        self.cid = cid
        self.info = {}

    @abstractmethod
    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""

    @abstractmethod
    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided weights using the locally held dataset."""

    @abstractmethod
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""

    @abstractmethod
    def federated_personalized_evaluate(self, ins: EvaluateIns) -> Tuple[EvaluateRes, EvaluateRes]:
        """FPE the provided weights using the locally held dataset."""

    @abstractmethod
    def reconnect(self, reconnect: Reconnect) -> Disconnect:
        """Disconnect and (optionally) reconnect later."""
