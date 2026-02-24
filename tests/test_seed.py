"""Tests for cube.seed - BasicSeedGenerator."""

from cube.seed import BasicSeedGenerator
from cube.task import TaskMetadata


def test_basic_seed_generator_returns_list_of_ints():
    seeds = BasicSeedGenerator(n_seed=3, meta_seed=42)(TaskMetadata(id="task-1"))
    assert isinstance(seeds, list)
    assert len(seeds) == 3
    assert all(isinstance(s, int) for s in seeds)


def test_basic_seed_generator_is_deterministic():
    gen = BasicSeedGenerator(n_seed=4, meta_seed=0)
    tm = TaskMetadata(id="repro-task")
    assert gen(tm) == gen(tm)


def test_basic_seed_generator_different_tasks_produce_different_seeds():
    gen = BasicSeedGenerator(n_seed=3, meta_seed=99)
    assert gen(TaskMetadata(id="task-alpha")) != gen(TaskMetadata(id="task-beta"))


def test_basic_seed_generator_different_meta_seeds_produce_different_seeds():
    tm = TaskMetadata(id="same-task")
    assert BasicSeedGenerator(n_seed=3, meta_seed=1)(tm) != BasicSeedGenerator(n_seed=3, meta_seed=2)(tm)


def test_basic_seed_generator_seeds_in_valid_numpy_range():
    for seed in BasicSeedGenerator(n_seed=10, meta_seed=1)(TaskMetadata(id="range-task")):
        assert 0 <= seed < 2**32
