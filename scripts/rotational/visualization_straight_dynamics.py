import numpy as np
import matplotlib.pyplot as plt

from vartools.dynamics import Dynamics
from vartools.dynamics import ConstantValue, LinearSystem
from vartools.dynamics import SinusAttractorSystem

from vartools.directional_space import fast_directional_vector_addition


def setup_plot(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_collinear_dynamics(n_grid=15, figsize=(3.0, 2.8), save_figure=False):
    x_lim = [-3.5, 3.5]
    y_lim = [-3, 3]

    dynamics = ConstantValue(np.array([1.0, 0]))

    nx = ny = n_grid
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros_like(positions)
    for pp in range(positions.shape[1]):
        velocities[:, pp] = dynamics.evaluate(positions[:, pp])

    fig, ax = plt.subplots(figsize=figsize)
    ax.quiver(
        positions[0, :],
        positions[1, :],
        velocities[0, :],
        velocities[1, :],
        # color="black",
        color="blue",
        # scale=0.1,
        # color="#414141",
        zorder=0,
    )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    setup_plot(ax=ax)

    if save_figure:
        figname = "global_collinear_dynamics"
        plt.savefig("figures/" + figname + filetype, bbox_inches="tight")


def plot_straight_dynamics(n_grid=15, figsize=(3.0, 2.8), save_figure=False):
    x_lim = [-3.5, 3.5]
    y_lim = [-3, 3]

    dynamics = LinearSystem(attractor_position=np.array([0.0, 0]), maximum_velocity=1.0)

    nx = ny = n_grid
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros_like(positions)
    for pp in range(positions.shape[1]):
        velocities[:, pp] = dynamics.evaluate(positions[:, pp])

    fig, ax = plt.subplots(figsize=figsize)
    ax.quiver(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        velocities[0, :].reshape(nx, ny),
        velocities[1, :].reshape(nx, ny),
        # color="black",
        color="blue",
        # scale=0.1,
        # color="#414141",
        zorder=0,
    )

    ax.plot(dynamics.attractor_position[0], dynamics.attractor_position[1], "k*")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    setup_plot(ax=ax)

    if save_figure:
        figname = "global_straight_dynamics"
        plt.savefig("figures/" + figname + filetype, bbox_inches="tight")


class LocallyStraightDynamics(Dynamics):
    def __init__(
        self, attractor_position=np.array([0, 0]), maximum_velocity=None
    ) -> None:

        self.linear = ConstantValue(np.array([1, 0.0]))
        self.wavy_dynamics = SinusAttractorSystem(attractor_position=attractor_position)

        self.center_straight = np.array([-2.0, 2.0])
        self.inner_radius = 0.5
        self.outer_radius = 1.0

        self.maximum_velocity = maximum_velocity
        self.attractor_position = attractor_position

    def get_weight(self, position) -> float:
        distance = np.linalg.norm(position - self.center_straight)

        if distance < self.inner_radius:
            return 1
        elif distance > self.outer_radius:
            return 0
        else:
            return (distance - self.inner_radius) / (
                self.outer_radius - self.inner_radius
            )

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        weight = self.get_weight(position)

        main_dynamics = self.wavy_dynamics.evaluate(position)
        if not (norm_velocity := np.linalg.norm(main_dynamics)):
            return main_dynamics

        if self.maximum_velocity is not None:
            norm_velocity = min(self.maximum_velocity, norm_velocity)

        if weight >= 1:
            linear_dynamics = self.linear.evaluate(position)
            return linear_dynamics / np.linalg.norm(main_dynamics) * norm_velocity

        elif weight <= 0:
            return main_dynamics

        # Assumption of relative
        linear_dynamics = self.linear.evaluate(position)
        final_dynamics = fast_directional_vector_addition(
            main_dynamics, linear_dynamics, weight=weight
        )

        return final_dynamics * norm_velocity


def plot_locally_straight(n_grid=15, figsize=(3.0, 2.8), save_figure=False):
    x_lim = [-6, 1.0]
    y_lim = [-1, 5]

    dynamics = LocallyStraightDynamics(
        attractor_position=np.array([0.0, -0]), maximum_velocity=1.0
    )

    nx = ny = n_grid
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros_like(positions)
    for pp in range(positions.shape[1]):
        velocities[:, pp] = dynamics.evaluate(positions[:, pp])

    fig, ax = plt.subplots(figsize=figsize)
    ax.quiver(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        velocities[0, :].reshape(nx, ny),
        velocities[1, :].reshape(nx, ny),
        # color="black",
        color="blue",
        # scale=0.1,
        # color="#414141",
        zorder=0,
    )

    ax.plot(dynamics.attractor_position[0], dynamics.attractor_position[1], "k*")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    setup_plot(ax=ax)

    if save_figure:
        figname = "global_straight_dynamics"
        plt.savefig("figures/" + figname + filetype, bbox_inches="tight")


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")

    filetype = ".pdf"
    # plot_collinear_dynamics(save_figure=False)
    # plot_straight_dynamics(save_figure=False)
    plot_locally_straight(save_figure=False)
