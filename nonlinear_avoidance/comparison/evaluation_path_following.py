from pathlib import Path
from dataclasses import dataclass

import math
import numpy as np
import pandas as pd


@dataclass
class TrajectoryEvaluator:
    data_file: str
    # data_path: "str" = "/home/lukas/Code/nonlinear_avoidance/comparison/data"
    root_path = Path("/home/lukas/Code/nonlinear_obstacle_avoidance")
    data_path = Path("nonlinear_avoidance/comparison/data/wavy_path_simulation")

    n_runs: int = 0

    def get_converged(self):
        self.df = pd.read_csv(self.root_path / self.data_path / self.data_file)
        return self.df[" converged"] > 0

    def run(self, converged):
        # data = np.loadtxt(
        #     self.root_path / self.data_path / self.data_file,
        #     delimiter=",",
        #     dtype=float,
        #     skiprows=1,
        # )
        if self.df is None:
            self.df = pd.read_csv(self.root_path / self.data_path / self.data_file)

        self.n_runs = self.df.shape[0]
        self.frac_converged = sum(self.df[" converged"] > 0) / self.n_runs
        df = self.df[converged]

        self.distance_mean = df[" distance"].mean()
        self.distance_var = df[" distance"].var()

        occupied = 1 - df[" free_fraction"]
        self.occupied_mean = np.mean(occupied)
        self.occupied_std = np.std(np.mean(occupied))

        self.gammas_mean = df[" gammas"].mean()
        self.gammas_var = df[" gammas"].var()


def get_converged(evaluator_list):
    converged = evaluator_list[0].get_converged()
    for evaluator in evaluator_list:
        converged = converged * evaluator.get_converged()

    return converged


def evaluate():
    evaluator_list = []

    evaluator_list.append(TrajectoryEvaluator("wavy_path_switching_straight.csv"))
    evaluator_list.append(TrajectoryEvaluator("wavy_path_switching_path.csv"))
    evaluator_list.append(TrajectoryEvaluator("wavy_path_global_nonlinear.csv"))

    indeces_converged = get_converged(evaluator_list)
    for evaluator in evaluator_list:
        evaluator.run(indeces_converged)

    return evaluator_list


def print_results(evaluation_list):
    value = [ee.data_file for ee in evaluation_list]
    print(" & ".join(["Name"] + value) + " \\\\ \hline")

    value = [f"{ee.frac_converged * 100:.0f}" for ee in evaluation_list]
    print(" & ".join(["Converged [\\%]"] + value) + " \\\\ \hline")

    value = [
        f"{ee.occupied_mean * 100:.2f}" + " $\\pm$ " + f"{ee.occupied_std * 100:.2f}"
        for ee in evaluation_list
    ]
    print(" & ".join(["Free [\\%]"] + value) + " \\\\ \hline")

    value = [
        f"{ee.gammas_mean:.2f}" + " $\\pm$ " + f"{ee.gammas_var:.2f}"
        for ee in evaluation_list
    ]
    print(" & ".join(["$\Delta d$"] + value) + " \\\\ \hline")

    value = [
        f"{ee.distance_mean:.2f}" + " $\\pm$ " + f"{ee.distance_var:.2f}"
        for ee in evaluation_list
    ]
    print(" & ".join(["Pathlength [m]"] + value) + " \\\\ \hline")


if (__name__) == "__main__":
    evaluation_list = evaluate()
    print_results(evaluation_list)
