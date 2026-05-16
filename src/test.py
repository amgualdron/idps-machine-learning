import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import hoomd
import hoomd.md
import hoomd.write
import numpy as np
import yaml

# ── Config loading ─────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
print(project_root)
