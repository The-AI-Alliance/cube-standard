import sys
from pathlib import Path

# Allow integration tests to import test_harness as a top-level module,
# matching the behaviour of running ``python tests/test_*.py`` directly.
sys.path.insert(0, str(Path(__file__).parent))
