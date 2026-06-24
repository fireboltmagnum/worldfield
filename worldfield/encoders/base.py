"""Abstract encoder interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    @abstractmethod
    def encode(self, input_data) -> np.ndarray:
        ...

    @abstractmethod
    def encode_batch(self, inputs: list) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...
