from pathlib import Path
from dataclasses import dataclass

import math
import numpy as np


class TrajectoryEvaluator:
    data_folder: "str"
    # data_path: "str" = "/home/lukas/Code/nonlinear_avoidance/comparison/data"
    data_path: Path = Path(
        "/home/lukas/Code/nonlinear_obstacle_avoidance",
        "nonlinear_avoidance/comparison/data/wavy_path_simulation/",
    )

    n_runs: int = 0

    def run(self):
        pass
