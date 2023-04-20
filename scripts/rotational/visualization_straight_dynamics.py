import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from vartools.dynamics import Dynamics
from vartools.dynamics import ConstantValue, LinearSystem
from vartools.dynamics import SinusAttractorSystem

from vartools.directional_space import fast_directional_vector_addition


def setup_plot(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_collinear_dynamics(n_grid=15, figsize=(3.0, 2.8), save_figure=False):
    x_lim = [-2.8, 2.8]
    y_lim = [-3.0, 3.0]

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
        width=0.009,
        scale=15,
    )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    setup_plot(ax=ax)

    if save_figure:
        figname = "global_collinear_dynamics"
        plt.savefig("figures/" + figname + filetype, bbox_inches="tight")


def plot_straight_dynamics(n_grid=15, figsize=(3.0, 2.8), save_figure=False):
    x_lim = [-2.8, 2.8]
    y_lim = [-3.0, 3.0]

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
        width=0.009,
        scale=15,
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

        self.center_straight = np.array([-2.0, 1.0])
        self.inner_radius = 1.0
        self.outer_radius = 2.0

        self.maximum_velocity = maximum_velocity
        self.attractor_position = attractor_position

    def get_weight(self, position) -> float:
        distance = np.linalg.norm(position - self.center_straight)

        if distance < self.inner_radius:
            return 1
        elif distance > self.outer_radius:
            return 0
        else:
            return (self.outer_radius - distance) / (
                self.outer_radius - self.inner_radius
            )

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        weight = self.get_weight(position)

        dist_attr = np.linalg.norm(position - self.attractor_position)
        if not dist_attr:
            return np.zeros_like(position)

        if self.maximum_velocity is None:
            norm_velocity = dist_attr
        else:
            norm_velocity = min(self.maximum_velocity, dist_attr)

        if weight >= 1:
            linear_dynamics = self.linear.evaluate(position)
            return linear_dynamics / np.linalg.norm(linear_dynamics) * norm_velocity

        elif weight <= 0:
            main_dynamics = self.wavy_dynamics.evaluate(position)
            return main_dynamics / np.linalg.norm(main_dynamics) * norm_velocity

        # Assumption of relative
        main_dynamics = self.wavy_dynamics.evaluate(position)
        main_dynamics = main_dynamics / np.linalg.norm(main_dynamics)
        linear_dynamics = self.linear.evaluate(position)
        linear_dynamics = linear_dynamics / np.linalg.norm(linear_dynamics)
        final_dynamics = fast_directional_vector_addition(
            main_dynamics, linear_dynamics, weight=weight
        )

        return final_dynamics * norm_velocity


def plot_locally_straight(n_grid=12, figsize=(3.0, 2.8), save_figure=False):
    x_lim = [-5, 0.6]
    y_lim = [-3.0, 3.0]

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
        width=0.009,
        scale=15,
    )

    # Re-evaluate for weight
    nx = ny = 100
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    weights = np.zeros(positions.shape[1])
    for pp in range(positions.shape[1]):
        weights[pp] = dynamics.get_weight(positions[:, pp])

    black_white_map = [[1, 1, 1], [0, 0, 0]]
    bw_cmap = ListedColormap(black_white_map)
    cont = ax.contourf(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        weights.reshape(nx, ny),
        # (weights >= 1).reshape(nx, ny),
        zorder=-2,
        # cmap=bw_cmap,
        # levels=np.linspace(0.0, 1.0, 3),
        cmap="Greys",
        levels=np.linspace(0.0, 1.0, 10),
        alpha=0.5,
    )

    ax.plot(dynamics.attractor_position[0], dynamics.attractor_position[1], "k*")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    setup_plot(ax=ax)

    if save_figure:
        figname = "locally_straight_dynamics"
        plt.savefig("figures/" + figname + filetype, bbox_inches="tight")


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")

    filetype = ".pdf"
    figsize = (2.8, 3.0)
    n_grid = 12
    plot_collinear_dynamics(save_figure=True, figsize=figsize, n_grid=n_grid)
    plot_straight_dynamics(save_figure=True, figsize=figsize, n_grid=n_grid)
    plot_locally_straight(save_figure=True, figsize=figsize, n_grid=n_grid)
