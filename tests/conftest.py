import sys
from pathlib import Path

# Make `import v2` / `import tools.v2_*` work regardless of how pytest is run.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
