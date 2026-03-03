"""
Seed generation utilities for CUBE.

This module provides deterministic seed generation for reproducible task randomization,
used to assign consistent random seeds to tasks based on their metadata.

Abstract classes:
    AbstractSeedGenerator — subclasses must implement:
        __call__(task_metadata: TaskMetadata) -> list[int]   return the list of
                                                             seeds for a given task
"""

import hashlib
import random
from abc import ABC, abstractmethod

from pydantic import BaseModel

from cube.task import TaskMetadata


class AbstractSeedGenerator(ABC, BaseModel):
    @abstractmethod
    def __call__(self, task_metadata: TaskMetadata) -> list[int]:
        """Given task metadata, return a list of seeds to use for that task."""
        pass


class BasicSeedGenerator(AbstractSeedGenerator):
    n_seed: int
    meta_seed: int

    def make_base_seed(self, text: str) -> int:
        # Combine text and meta_seed into a single string
        combined = f"{text}_{self.meta_seed}"
        # Hash it to get a consistent number
        hash_obj = hashlib.md5(combined.encode())
        # Convert hash to integer and take modulo to fit seed range
        return int(hash_obj.hexdigest(), 16) % (2**32)

    def __call__(self, task_metadata: TaskMetadata) -> list[int]:
        base_seed = self.make_base_seed(task_metadata.id)
        rng = random.Random(base_seed)
        return [rng.randrange(2**32) for _ in range(self.n_seed)]
