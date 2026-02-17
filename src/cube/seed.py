import hashlib
from abc import abstractmethod

import numpy as np

from cube.task import TaskMetadata


class AbstractSeedGenerator:
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
        # Convert hash to integer and take modulo to fit numpy's seed range
        return int(hash_obj.hexdigest(), 16) % (2**32)

    def __call__(self, task_metadata: TaskMetadata) -> list[int]:
        base_seed = self.make_base_seed(task_metadata.id)
        rng = np.random.default_rng(base_seed)
        return rng.integers(0, 2**32, size=self.n_seed).tolist()
