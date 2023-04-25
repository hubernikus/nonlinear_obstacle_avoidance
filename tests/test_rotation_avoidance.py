#!/USSR/bin/python3.9
""" Test overrotation for ellipses. """

# Created: 2021-08-04
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import warnings
from functools import partial
import unittest

# from math import pi
import math

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.linalg import get_orthogonal_basis
from vartools.dynamical_systems import LinearSystem, ConstantValue
from vartools.dynamical_systems import QuadraticAxisConvergence
from vartools.dynamical_systems import CircularStable
from vartools.directional_space import UnitDirection
from vartools.states import ObjectPose

# DirectionBase
from vartools.dynamical_systems import plot_vectorfield
from vartools.math import get_intersection_with_circle

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import StarshapedFlower
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.dynamics.circular_dynamics import SimpleCircularDynamics
from nonlinear_avoidance.dynamics import WavyLinearDynamics
from nonlinear_avoidance.rotation_container import RotationContainer
from nonlinear_avoidance.avoidance import obstacle_avoidance_rotational
from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)

from dynamic_obstacle_avoidance.visualization import Simulation_vectorFields

# from dynamic_obstacle_avoidance.visualization import plot_obstacles


def test_intersection_with_circle():
    # One Dimensional Circle
    start_position = np.array([0.1])
    radius = 2.1
    direction = np.array([-2])

    direction = direction / LA.norm(direction)
    circle_position = get_intersection_with_circle(
        start_position=start_position,
        direction=direction,
        radius=radius,
    )

    dir_new = circle_position - start_position
    dir_new = dir_new / LA.norm(dir_new)
    assert np.allclose(dir_new, direction)
    assert np.isclose(LA.norm(circle_position), radius)

    # Two Dimensional Circle
    start_position = np.array([0.3, 0.5])
    radius = 1.4
    direction = np.array([3, 1])

    direction = direction / LA.norm(direction)
    circle_position = get_intersection_with_circle(
        start_position=start_position,
        direction=direction,
        radius=radius,
    )

    dir_new = circle_position - start_position
    dir_new = dir_new / LA.norm(dir_new)
    assert np.allclose(dir_new, direction)
    assert np.isclose(LA.norm(circle_position), radius)

    # 1 Dimensional Circle with 2 Points
    # One Dimensional Circle
    start_position = np.array([0.1])
    radius = 2.1
    direction = np.array([-2])

    direction = direction / LA.norm(direction)
    circle_positions = get_intersection_with_circle(
        start_position=start_position,
        direction=direction,
        radius=radius,
        only_positive=False,
    )

    dir_new = circle_positions[:, 0] - start_position
    dir_new = dir_new / LA.norm(dir_new)
    assert np.allclose(dir_new, (-1) * direction)

    dir_new = circle_positions[:, 1] - start_position
    dir_new = dir_new / LA.norm(dir_new)
    assert np.allclose(dir_new, direction)

    # Points are both on boundary
    assert np.isclose(LA.norm(circle_positions[:, 0]), radius)
    assert np.isclose(LA.norm(circle_positions[:, 1]), radius)


def old_test_rotational_pulling(visualize=False):
    # Testing the non-linear 'pulling' (based on linear velocity)
    # nonlinear_velocity = np.array([1, 0])

    normal = (-1) * np.array([-1, -1])
    base = get_orthogonal_basis(vector=normal)

    dir_nonlinear = UnitDirection(base).from_vector(np.array([1, 0]))
    convergence_dir = UnitDirection(base).from_vector(np.array([0, 1]))

    inv_nonlinear = dir_nonlinear.invert_normal()
    inv_conv_rotated = convergence_dir.invert_normal()

    main_avoider = RotationalAvoider()
    inv_conv_proj = main_avoider._get_projection_of_inverted_convergence_direction(
        inv_conv_rotated=inv_conv_rotated,
        inv_nonlinear=inv_nonlinear,
        inv_convergence_radius=math.pi / 2,
    )

    assert inv_nonlinear.as_angle() < inv_conv_proj.as_angle(), " Not rotated enough."

    assert inv_conv_proj.as_angle() < inv_conv_rotated.as_angle(), " Rotated too much."

    nonlinear_conv = main_avoider._get_projected_nonlinear_velocity(
        dir_conv_rotated=convergence_dir,
        dir_nonlinear=dir_nonlinear,
        convergence_radius=math.pi / 2,
        weight=0.5,
    )

    if visualize:
        # Inverted space
        fig, ax = plt.subplots()
        ax.set_title("Inverted Directions")
        ax.plot([-3.5, 3.5], [0, 0], "k--")
        ax.plot([-math.pi, math.pi], [0, 0], color="red")
        ax.plot([-math.pi / 2, math.pi / 2], [0, 0], color="green")
        ax.plot([-math.pi, 0, math.pi], [0, 0, 0], "|", color="black")

        ax.plot(inv_nonlinear.as_angle(), 0, "o", color="blue", label="Nonlinear")
        ax.plot(
            inv_conv_rotated.as_angle(), 0, "o", color="darkviolet", label="Convergence"
        )
        ax.plot(inv_conv_proj.as_angle(), 0, "x", color="darkorange", label="Projected")

        ax.legend()

        # Plot with normal at center
        fig, ax = plt.subplots()
        ax.set_title("General Directions")
        ax.plot([-3.5, 3.5], [0, 0], "k--")
        ax.plot([-math.pi, math.pi], [0, 0], color="green")
        ax.plot([-math.pi / 2, math.pi / 2], [0, 0], color="red")
        ax.plot([-math.pi, 0, math.pi], [0, 0, 0], "|", color="black")

        ax.plot(dir_nonlinear.as_angle(), 0, "o", color="blue", label="Nonlinear")
        ax.plot(
            convergence_dir.as_angle(), 0, "o", color="darkviolet", label="Convergence"
        )
        ax.plot(nonlinear_conv.as_angle(), 0, "x", color="darkorange", label="Rotated")

        ax.set_xlim([-3.5, 3.5])
        ax.set_ylim([-1, 1])

        ax.legend()

    # The velocity needs to be in between
    assert (
        np.cross(dir_nonlinear.as_vector(), nonlinear_conv.as_vector()) >= 0
    ), " Not rotated enough."

    # The velocity needs to be in between
    assert (
        np.cross(convergence_dir.as_vector(), nonlinear_conv.as_vector()) <= 0
    ), "Rotated too much."


def test_convergence_tangent(visualize=True):
    initial_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))

    # ConvergingDynamics=ConstantValue (initial_velocity)
    obstacle = Ellipse(
        center_position=np.array([0, 0]),
        axes_length=np.array([1, 1]),
    )

    position = np.array([-1, 1])
    linear_velocity = initial_dynamics.evaluate(position)

    normal = obstacle.get_normal_direction(position, in_global_frame=True)
    normal_base = get_orthogonal_basis(normal * (-1))

    delta_pos = obstacle.center_position - position
    convergence_radius = math.pi / 2.0
    main_avoider = RotationalAvoider()
    tangent = main_avoider.get_tangent_convergence_direction(
        dir_convergence=UnitDirection(normal_base).from_vector(linear_velocity),
        dir_reference=UnitDirection(normal_base).from_vector(delta_pos),
        # base=normal_base,
        convergence_radius=convergence_radius,
    )

    assert np.allclose(
        tangent.as_vector(), np.sqrt(2) / 2 * np.array([1, 1])
    ), " Not rotated enough."

    # Same thing for alternative tangent
    tangent_vector = RotationalAvoider.get_projected_tangent_from_vectors(
        initial_vector=linear_velocity,
        normal=normal,
        reference=delta_pos,
        convergence_radius=convergence_radius,
    )
    assert np.allclose(
        tangent_vector, np.sqrt(2) / 2 * np.array([1, 1])
    ), " Not rotated enough."

    if visualize:
        obstacle = Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([1, 2]),
        )
        fig, ax = plt.subplots()

        x_lim = [-10, 10]
        y_lim = [-10, 10]

        nx = ny = 20
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )

        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        vectors = np.zeros(positions.shape)
        for it in range(positions.shape[1]):
            linear_velocity = initial_dynamics.evaluate(position)

            normal = obstacle.get_normal_direction(
                positions[:, it], in_global_frame=True
            )
            normal_base = get_orthogonal_basis(normal * (-1))
            delta_dir = obstacle.center_position - positions[:, it]
            unit_tangent = main_avoider.get_tangent_convergence_direction(
                dir_convergence=UnitDirection(normal_base).from_vector(linear_velocity),
                dir_reference=UnitDirection(normal_base).from_vector(delta_dir),
                # base=normal_base,
                convergence_radius=math.pi / 2,
            )

            vectors[:, it] = unit_tangent.as_vector()

        ax.quiver(
            positions[0, :],
            positions[1, :],
            vectors[0, :],
            vectors[1, :],
            color="blue",
        )

        ax.set_aspect("equal", adjustable="box")


def test_rotating_towards_tangent():
    initial_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))

    # ConvergingDynamics=ConstantValue (initial_velocity)
    obstacle = Ellipse(
        center_position=np.array([0, 0]),
        axes_length=np.array([1, 1]),
    )

    position = np.array([-1, 1])
    linear_velocity = initial_dynamics.evaluate(position)

    normal = obstacle.get_normal_direction(position, in_global_frame=True)
    normal_base = get_orthogonal_basis(normal * (-1))

    delta_dir = obstacle.center_position - position
    main_avoider = RotationalAvoider()
    tangent = main_avoider.get_tangent_convergence_direction(
        dir_convergence=UnitDirection(normal_base).from_vector(linear_velocity),
        dir_reference=UnitDirection(normal_base).from_vector(delta_dir),
        # base=normal_base,
        convergence_radius=math.pi / 2,
    )

    rotated_velocity = main_avoider._get_projected_velocity(
        dir_convergence_tangent=tangent,
        dir_initial_velocity=UnitDirection(normal_base).from_vector(linear_velocity),
        weight=0.5,
        convergence_radius=math.pi / 2,
    )

    assert (
        np.cross(linear_velocity, rotated_velocity.as_vector()) > 0
    ), " Not rotated enough."

    assert (
        np.cross(rotated_velocity.as_vector(), tangent.as_vector()) > 0
    ), " Rotated too much."


def test_rotated_convergence_direction_circle():
    initial_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))

    # ConvergingDynamics=ConstantValue (initial_velocity)
    obstacle = Ellipse(
        center_position=np.array([0, 0]),
        axes_length=np.array([1, 1]),
    )

    weight = 0.5
    position = np.array([-1.0, 0.5])

    inital_velocity = initial_dynamics.evaluate(position)

    normal = obstacle.get_normal_direction(position, in_global_frame=True)
    norm_base = get_orthogonal_basis(normal * (-1))

    main_avoider = RotationalAvoider()
    convergence_dir = main_avoider._get_rotated_convergence_direction(
        weight=weight,
        convergence_radius=math.pi / 2.0,
        convergence_vector=inital_velocity,
        reference_vector=obstacle.get_reference_direction(
            position, in_global_frame=True
        ),
        base=norm_base,
    )

    initial_dir = UnitDirection(norm_base).from_vector(inital_velocity)

    assert (
        convergence_dir.norm() > initial_dir.norm()
    ), "Rotational convergence not further away from norm."

    assert (
        np.cross(inital_velocity, convergence_dir.as_vector()) > 0
    ), "Rotation in the wrong direction."


def test_rotated_convergence_direction_ellipse():
    initial_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))

    obstacle = Ellipse(
        center_position=np.array([0, 0]),
        axes_length=np.array([1, 2]),
    )

    weight = 0.5
    position = np.array([-1.0, 0.5])

    inital_velocity = initial_dynamics.evaluate(position)

    normal = obstacle.get_normal_direction(position, in_global_frame=True)
    norm_base = get_orthogonal_basis(normal * (-1))

    main_avoider = RotationalAvoider()
    convergence_dir = main_avoider._get_rotated_convergence_direction(
        weight=weight,
        convergence_radius=math.pi / 2.0,
        convergence_vector=inital_velocity,
        reference_vector=obstacle.get_reference_direction(
            position, in_global_frame=True
        ),
        base=norm_base,
    )

    initial_dir = UnitDirection(norm_base).from_vector(inital_velocity)

    assert (
        convergence_dir.norm() > initial_dir.norm()
    ), "Rotational convergence not further away from norm."

    assert (
        np.cross(inital_velocity, convergence_dir.as_vector()) > 0
    ), "Rotation in the wrong direction."


def test_single_circle_linear(visualize=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2, 2]),
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))

    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)
    # ConvergingDynamics=ConstantValue (initial_velocity)

    main_avoider = RotationalAvoider()

    if visualize:
        # Plot Normals
        Simulation_vectorFields(
            x_lim=[-2, 2],
            y_lim=[-2, 2],
            n_resolution=40,
            obstacle_list=obstacle_list,
            noTicks=False,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=initial_dynamics.evaluate,
            obs_avoidance_func=main_avoider.avoid,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            # Quiver or stream plot
            show_streamplot=False,
            # show_streamplot=False,
        )

    # No effect when already pointing away (save circle)
    position = np.array([1.12, 0.11])
    modulated_velocity = obstacle_avoidance_rotational(
        position,
        initial_dynamics.evaluate(position),
        obstacle_list,
    )
    # breakpoint()
    assert np.allclose(
        initial_dynamics.evaluate(position), modulated_velocity
    ), "Unexpected modulation behind the obstacle."

    # Decreasing influence with decreasing distance
    position = np.array([-1, 0.1])
    mod_vel = obstacle_avoidance_rotational(
        position, initial_dynamics.evaluate(position), obstacle_list
    )
    mod_vel1 = mod_vel / LA.norm(mod_vel)

    position = np.array([-2, 0.1])
    mod_vel = obstacle_avoidance_rotational(
        position,
        initial_dynamics.evaluate(position),
        obstacle_list
        # position, initial_velocity, obstacle_list
    )
    mod_vel2 = mod_vel / LA.norm(mod_vel)

    # Decreasing influence -> closer to 0 [without magnitude]
    velocity = initial_dynamics.evaluate(position)

    assert np.dot(mod_vel1, velocity) < np.dot(mod_vel2, velocity)

    # Velocity on surface is tangent after modulation
    rad_x = np.sqrt(2) / 2 + 1e-9
    position = np.array([(-1) * rad_x, rad_x])
    modulated_velocity = obstacle_avoidance_rotational(
        position, initial_dynamics.evaluate(position), obstacle_list
    )
    normal_dir = obstacle_list[0].get_normal_direction(position, in_global_frame=True)
    assert abs(np.dot(modulated_velocity, normal_dir)) < 1e-6

    # Point far away has no/little influence
    position = np.array([1e10, 0])
    modulated_velocity = obstacle_avoidance_rotational(
        position,
        initial_dynamics.evaluate(position),
        obstacle_list,
    )
    assert np.allclose(initial_dynamics.evaluate(position), modulated_velocity)

    # Rotate to the left on top
    position = np.array([-1, 0.5])
    initial_velocity = initial_dynamics.evaluate(position)

    modulated_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_list,
    )
    assert (
        np.cross(initial_velocity, modulated_velocity) > 0
    ), " Rotation in the wrong direction to avoid the circle."

    # Rotate to the right bellow
    position = np.array([-1, -0.5])
    initial_velocity = initial_dynamics.evaluate(position)

    modulated_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_list,
    )
    assert (
        np.cross(initial_velocity, modulated_velocity) < 0
    ), " Rotation in the wrong direction to avoid the circle."


def test_single_circle_linear_repulsive(visualize=False, save_figure=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2, 2]),
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([2.5, 0]))
    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)
    # ConvergingDynamics=ConstantValue (initial_velocity)
    main_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        convergence_radius=math.pi,
    )

    # main_avoider = partial(RotationalAvoider().avoid, convergence_radius=math.pi)

    if visualize:
        # Arbitrary constant velocity
        tmp_dynamics = LinearSystem(attractor_position=np.array([2.0, 0]))
        tmp_dynamics.distance_decrease = 0.1
        obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)
        # ConvergingDynamics=ConstantValue (initial_velocity)
        tmp_avoider = RotationalAvoider(
            initial_dynamics=initial_dynamics,
            obstacle_environment=obstacle_list,
            convergence_radius=math.pi,
        )
        x_lim = [-2, 3]
        y_lim = [-2.2, 2.2]
        n_grid = 13
        alpha_obstacle = 1.0

        plt.close("all")

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacle_dynamics(
            obstacle_container=obstacle_list,
            dynamics=tmp_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.attractor_position,
            do_quiver=True,
            show_ticks=False,
        )

        plot_obstacles(
            obstacle_container=obstacle_list,
            ax=ax,
            alpha_obstacle=alpha_obstacle,
        )

        if save_figure:
            fig_name = "circular_repulsion_pi"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)

    # Test that goes below in front
    position = np.array([-1, -1])
    initial_velocity = initial_dynamics.evaluate(position)
    modulated_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_list,
    )
    assert np.cross(initial_velocity, modulated_velocity) < 0

    # Test that goes below in front
    position = np.array([-1, 1])
    initial_velocity = initial_dynamics.evaluate(position)
    modulated_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_list,
    )
    assert np.cross(initial_velocity, modulated_velocity) > 0

    # Pointing away on surface
    position = np.array([1, 1]) * 1.0 / math.sqrt(2)
    initial_velocity = initial_dynamics.evaluate(position)
    modulated_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_list,
    )
    assert np.allclose(
        modulated_velocity / LA.norm(modulated_velocity), position / LA.norm(position)
    ), "Modulated velocity is expected to point away from the obstacle."

    # Pointing away on surface
    position = np.array([1, -1]) * 1.0 / math.sqrt(2)
    initial_velocity = initial_dynamics.evaluate(position)
    modulated_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_list,
    )

    assert np.allclose(
        modulated_velocity / LA.norm(modulated_velocity), position / LA.norm(position)
    ), "Modulated velocity is expected to point away from the obstacle."


def test_single_circle_linear_inverted(visualize=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 3]),
            is_boundary=True,
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([1, 0]))

    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)
    # ConvergingDynamics=ConstantValue (initial_velocity)

    main_avoider = RotationalAvoider()
    my_avoider = partial(main_avoider.avoid, convergence_radius=math.pi)

    # Little effect at center
    position = np.array([0, 1])
    modulated_velocity = obstacle_avoidance_rotational(
        position,
        initial_dynamics.evaluate(position),
        obstacle_list,
        convergence_radius=math.pi,
    )
    # assert modulated_velocity[0] > 1
    # assert modulated_velocity[1] == 0

    if visualize:
        # Plot Normals
        Simulation_vectorFields(
            x_lim=[-2, 2],
            y_lim=[-2, 2],
            n_resolution=20,
            obstacle_list=obstacle_list,
            noTicks=False,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=initial_dynamics.evaluate,
            obs_avoidance_func=my_avoider,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            # Quiver or stream plot
            show_streamplot=False,
            # show_streamplot=False,
        )


def test_single_perpendicular_ellipse(visualize=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([1, 2]),
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))

    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)
    # ConvergingDynamics=ConstantValue (initial_velocity)

    main_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics, obstacle_environment=obstacle_list
    )

    if visualize:
        Simulation_vectorFields(
            x_lim=[-2, 2],
            y_lim=[-2, 2],
            n_resolution=20,
            obstacle_list=obstacle_list,
            noTicks=False,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=initial_dynamics.evaluate,
            obs_avoidance_func=main_avoider.avoid,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            # Quiver or Streamplot
            show_streamplot=False,
            # show_streamplot=False,
        )
    position = np.array([-1, 0.5])
    initial_velocity = initial_dynamics.evaluate(position)

    # assert np.cross(initial_velocity, modulated_velocity) > 0, \
    # " Rotation in the wrong direction to avoid the ellipse."

    print("<< Ellipse >>")
    modulated_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_list,
    )
    print("Velocities: ")
    print(initial_velocity / LA.norm(initial_velocity))
    print(modulated_velocity / LA.norm(modulated_velocity))

    obstacle_list[-1].axes_length = np.array([1, 1])
    print("<< Circular >>")
    modulated_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_list,
    )
    print("Velocities: ")
    print(initial_velocity / LA.norm(initial_velocity))
    print(modulated_velocity / LA.norm(modulated_velocity))


def _test_single_circle_nonlinear(visualize=False, save_figure=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([-4, 0]),
            axes_length=np.array([6, 9]),
            distance_scaling=0.5,
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = QuadraticAxisConvergence(attractor_position=np.array([1.5, 0]))
    convergence_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))

    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)
    # ConvergingDynamics=ConstantValue (initial_velocity)

    main_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_list,
    )

    if visualize:
        # x_lim = [-4, 3]
        # y_lim = [-3, 3]

        x_lim = [-15, 3]
        y_lim = [-6, 9]

        n_grid = 50
        plt.close("all")

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=initial_dynamics.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.attractor_position,
            do_quiver=False,
            show_ticks=False,
            vectorfield_color="black",
        )

        plot_obstacles(
            obstacle_container=obstacle_list,
            ax=ax,
            alpha_obstacle=0.6,
        )

        if save_figure:
            fig_name = "rotation_avoidance_ellipse_initial"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacle_dynamics(
            obstacle_container=obstacle_list,
            dynamics=main_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.attractor_position,
            do_quiver=False,
            show_ticks=False,
            vectorfield_color="blue",
        )

        plot_obstacles(
            obstacle_container=obstacle_list,
            ax=ax,
            alpha_obstacle=1.0,
        )

        if save_figure:
            fig_name = "rotation_avoidance_ellipse"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)


def test_double_ellipse(visualize=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([-2, 0]),
            axes_length=np.array([1, 2]),
            name="center_ellipse",
        )
    )

    obstacle_list.append(
        Ellipse(
            center_position=np.array([1, 0]),
            axes_length=np.array([4, 2]),
            orientation=30 / 180.0 * math.pi,
        )
    )

    # Arbitrary constant velocity
    initial_velocity = np.array([1, 1])

    obstacle_list.set_convergence_directions(
        converging_dynamics=ConstantValue(initial_velocity)
    )

    if visualize:
        # Plot Normals
        Simulation_vectorFields(
            x_lim=[-4, 4],
            y_lim=[-4, 4],
            n_resolution=20,
            obstacle_list=obstacle_list,
            noTicks=False,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=lambda x: initial_velocity,
            obs_avoidance_func=obstacle_avoidance_rotational,
            automatic_reference_point=False,
            # pos_attractor=initial_dynamics.attractor_position,
            # Quiver or Streamplot
            show_streamplot=False,
            # show_streamplot=False,
        )

    # Random evaluation
    position = np.array([-4, 2])
    modulated_velocity = obstacle_avoidance_rotational(
        position, initial_velocity, obstacle_list
    )

    # Normal in either case
    position = np.array([-1, 0])
    modulated_velocity = obstacle_avoidance_rotational(
        position, initial_velocity, obstacle_list
    )

    normal_dir = obstacle_list[0].get_normal_direction(position, in_global_frame=True)
    # assert np.isclose(np.dot(modulated_velocity, normal_dir), 0)


def test_stable_linear_avoidance(visualize=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2, 2]),
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))
    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)

    if visualize:
        # Plot Normals
        Simulation_vectorFields(
            x_lim=[-2, 2],
            y_lim=[-2, 2],
            n_resolution=20,
            obstacle_list=obstacle_list,
            noTicks=False,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=initial_dynamics.evaluate,
            obs_avoidance_func=obstacle_avoidance_rotational,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            # Quiver or stream-plot
            show_streamplot=False,
            # show_streamplot=False,
        )


def _test_obstacle_and_hull_avoidance(visualize=False, save_figure=False):
    attractor_position = np.array([3.0, 2.0])
    # initial_dynamics = LinearSystem(attractor_position=attractor_position)
    initial_dynamics = WavyLinearDynamics(attractor_position=attractor_position)
    convergence_dynamics = LinearSystem(attractor_position=attractor_position)

    rotation_container = RotationContainer()
    rotation_container.set_convergence_directions(converging_dynamics=initial_dynamics)

    # rotation_container.append(
    #     Ellipse(
    #         center_position=np.array([1, -2]),
    #         axes_length=np.array([4, 2]),
    #         orientation=30 / 90.0 * math.pi,
    #     )
    # )

    rotation_container.append(
        StarshapedFlower(
            center_position=np.array([2, -2]),
            radius_magnitude=0.5,
            radius_mean=1.5,
            number_of_edges=3,
            # axes_length=np.array([4, 2]),
            orientation=30 / 90.0 * math.pi,
            distance_scaling=0.5,
        )
    )

    rotation_container.append(
        Cuboid(
            center_position=np.array([-2.5, 2]),
            axes_length=np.array([3, 1.5]),
            orientation=120 / 90.0 * math.pi,
            distance_scaling=0.5,
        )
    )

    rotation_container.append(
        Cuboid(
            center_position=np.array([0, 0]),
            axes_length=np.array([12, 9]),
            is_boundary=True,
            distance_scaling=3,
        )
    )

    rotation_avoider = RotationalAvoider(
        obstacle_environment=rotation_container,
        # convergence_radius=math.pi / 2.0,
        initial_dynamics=initial_dynamics,
    )

    if visualize:
        x_lim = [-6.5, 6.5]
        y_lim = [-5.0, 5.0]
        n_grid = 120
        fig, ax = plt.subplots(figsize=(7, 5))

        plot_obstacle_dynamics(
            obstacle_container=rotation_container,
            dynamics=rotation_avoider.avoid,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.attractor_position,
            do_quiver=False,
            show_ticks=False,
        )

        plot_obstacles(
            obstacle_container=rotation_container,
            ax=ax,
            alpha_obstacle=1.0,
            x_lim=x_lim,
            y_lim=y_lim,
        )

        if save_figure:
            fig_name = "avoidance_obstacles_in_hull"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight")


def test_tangent_finding():
    normal = np.array([-0.1618981, -0.98680748])
    reference = np.array([0.81478719, 0.57976015])

    initial = np.array([0.98680748, -0.1618981])

    projected = RotationalAvoider.get_projected_tangent_from_vectors(
        initial,
        normal=normal,
        reference=reference,
    )
    assert projected[0] > 0 and projected[1] < 0


def test_simple_convergence_in_limit_cycle():
    obstacle_environment = RotationContainer()
    center = np.array([2.2, 0.0])
    obstacle_environment.append(
        StarshapedFlower(
            center_position=center,
            radius_magnitude=0.3,
            number_of_edges=4,
            radius_mean=0.75,
            orientation=10 / 180 * math.pi,
            distance_scaling=1,
            # is_boundary=True,
        )
    )

    attractor_position = np.array([0.0, 0])
    circular_ds = SimpleCircularDynamics(
        radius=2.0,
        pose=ObjectPose(
            position=attractor_position,
        ),
    )

    # Approximate limit cycle with circular.
    convergence_dynamics = LinearSystem(attractor_position=attractor_position)
    obstacle_avoider_globally_straight = RotationalAvoider(
        initial_dynamics=circular_ds,
        obstacle_environment=obstacle_environment,
        convergence_system=convergence_dynamics,
    )

    # Move continuously up
    position1 = np.array([1.2734, -1.0926])
    velocity1 = obstacle_avoider_globally_straight.evaluate(position1)
    assert velocity1[1] > 0

    # A close position should have close velocity values
    position2 = np.array([1.2703, -1.1460])
    velocity2 = obstacle_avoider_globally_straight.evaluate(position2)
    assert np.allclose(velocity1, velocity2, atol=1e-1)


def test_axes_following_rotation(visualize=False, x_lim=[-4, 4], y_lim=[-4, 4]):
    attractor = np.array([8.0, 0])

    initial_dynamics = QuadraticAxisConvergence(
        stretching_factor=10,
        maximum_velocity=1.0,
        dimension=2,
        attractor_position=attractor,
    )

    environment = RotationContainer()
    environment.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 5]),
            orientation=0.0 / 180 * math.pi,
            is_boundary=False,
            tail_effect=False,
            distance_scaling=0.5,
        )
    )

    avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=environment,
        convergence_system=LinearSystem(attractor)
        # convergence_radius=math.pi * 0.7,
    )

    if visualize:
        plt.close("all")
        fig, ax = plt.subplots(figsize=(6, 5))

        plot_obstacles(
            ax=ax,
            obstacle_container=environment,
            x_range=x_lim,
            y_range=y_lim,
            # noTicks=True,
            showLabel=False,
            alpha_obstacle=0.9,
        )

        plot_vectorfield = True
        if plot_vectorfield:
            plot_obstacle_dynamics(
                obstacle_container=environment,
                collision_check_functor=lambda x: (
                    environment.get_minimum_gamma(x) <= 1
                ),
                dynamics=avoider.evaluate,
                attractor_position=initial_dynamics.attractor_position,
                x_lim=x_lim,
                y_lim=y_lim,
                ax=ax,
                do_quiver=False,
                show_ticks=True,
            )

    # Just in front towards the top of the ellipse
    position = np.array([-1.28, 1.82])
    rotated_dynamics = avoider.avoid(position)
    assert (
        rotated_dynamics[0] > 0 and rotated_dynamics[1] > 0
    ), "Expected to avoid towards top-right."

    # Just in front, towards the center of the ellipse
    position = np.array([-1.74, 0.55])
    rotated_dynamics = avoider.avoid(position)
    assert (
        rotated_dynamics[0] > 0 and rotated_dynamics[1] > 0
    ), "Expected to avoid towards top-right."


def test_overrotation_starshape():
    obstacle_list = RotationContainer()
    obstacle_list.append(
        StarshapedFlower(
            center_position=np.zeros(2),
            number_of_edges=3,
            radius_magnitude=0.2,
            radius_mean=0.75,
            orientation=10 / 180 * math.pi,
            distance_scaling=1,
            # is_boundary=True,
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([2.5, 0]))
    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)

    main_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        convergence_radius=math.pi,
    )

    # Arbitrary constant velocity
    tmp_dynamics = LinearSystem(
        attractor_position=np.array([2.0, 0]), maximum_velocity=1.0
    )
    tmp_dynamics.distance_decrease = 0.1
    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)
    # ConvergingDynamics=ConstantValue (initial_velocity)

    angle = math.pi
    obstacle_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        convergence_radius=angle,
    )

    position = np.array([-1.16, 0.36])
    print("position", position)
    velocity = obstacle_avoider.evaluate(position)
    assert velocity[1] > 0, "Velocity is required to move away from saddle point."

    position = np.array([-0.739, 0.370])
    velocity = obstacle_avoider.evaluate(position)
    assert velocity[0] < 0, "Full repulsion on the surface."

    angle = math.pi / 2.0
    obstacle_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        convergence_radius=angle,
    )

    position = np.array([-1.16, 0.36])
    velocity = obstacle_avoider.evaluate(position)
    assert velocity[1] > 0, "Velocity is required to move away from saddle point."


if (__name__) == "__main__":
    figtype = ".pdf"
    # figtype = ".png"

    test_overrotation_starshape()

    # test_intersection_with_circle()

    # test_convergence_tangent(visualize=False)
    # test_rotating_towards_tangent()

    # test_single_circle_linear(visualize=True)
    # test_single_circle_linear_inverted(visualize=True)

    # _test_single_circle_nonlinear(visualize=True, save_figure=True)
    test_single_circle_linear_repulsive(visualize=True, save_figure=False)

    # test_rotated_convergence_direction_circle()
    # test_rotated_convergence_direction_ellipse()

    # test_single_perpendicular_ellipse(visualize=True)

    # test_double_ellipse(visualize=True)
    # test_stable_linear_avoidance(visualize=True)

    # _test_obstacle_and_hull_avoidance(visualize=True, save_figure=True)
    # test_simple_convergence_in_limit_cycle()
    # test_axes_following_rotation(visualize=False)

    print("[Rotational Tests] Done tests")
