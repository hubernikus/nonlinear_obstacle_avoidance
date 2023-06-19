#!/USSR/bin/python3
"""
Create the rotation space which is so much needed.
"""
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-07-07

import math

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from nonlinear_avoidance.vector_rotation import VectorRotationXd
from nonlinear_avoidance.vector_rotation import VectorRotationTree
from nonlinear_avoidance.vector_rotation import VectorRotationSequence

from nonlinear_avoidance.vector_rotation import rotate_direction


def test_cross_rotation_2d(visualize=False, savefig=False):
    vec0 = np.array([1, 0.3])
    vec1 = np.array([1.0, -1])

    vector_rotation = VectorRotationXd.from_directions(vec0, vec1)

    vec0 /= LA.norm(vec0)
    vec1 /= LA.norm(vec1)

    # Reconstruct vector1
    vec_rot = vector_rotation.rotate(vec0)
    assert np.allclose(vec1, vec_rot), "Rotation was not reconstructed."

    vecs_test = [
        [1, -1.2],
        [-1.2, -1],
        [-1.2, 1.3],
    ]

    cross_prod_base = np.cross(vec0, vec1)

    vecs_rot_list = []
    for ii, vec in enumerate(vecs_test):
        vec_test = np.array(vecs_test[ii])
        vec_test /= LA.norm(vec_test)
        vec_rot = vector_rotation.rotate(vec_test)

        assert np.isclose(
            cross_prod_base, np.cross(vec_test, vec_rot)
        ), "Vectors are not close"

        # assert np.isclose(
        #     vector_rotation.rotation_angle, np.arccos(np.dot(vec_test, vec_rot))
        # ), "Not the correct rotation."

        # For visualization purposes
        vecs_rot_list.append(vec_rot)

    if visualize:
        arrow_props = {"head_length": 0.1, "head_width": 0.05}

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        ax = axs[0, 0]
        vec_1 = vector_rotation.base[:, 0]
        ax.arrow(
            0,
            0,
            vec_1[0],
            vec_1[1],
            color="k",
            **arrow_props,
        )

        vec_perp = vector_rotation.base[:, 1]
        ax.arrow(
            0,
            0,
            vec_perp[0],
            vec_perp[1],
            color="k",
            label="Base",
            **arrow_props,
        )

        ax.arrow(0, 0, vec0[0], vec0[1], color="g", label="Vector 0", **arrow_props)
        ax.arrow(0, 0, vec1[0], vec1[1], color="b", label="Vector 1", **arrow_props)
        ax.legend()

        ax = axs[0, 1]
        axs_test = axs.flatten()[1:]
        for ii, ax in enumerate(axs_test):
            vec_test = vecs_test[ii] / LA.norm(vecs_test[ii])
            ax.arrow(
                0,
                0,
                vec_test[0],
                vec_test[1],
                color="g",
                label="Initial",
                **arrow_props,
            )
            vec_rot = vecs_rot_list[ii]

            ax.arrow(
                0, 0, vec_rot[0], vec_rot[1], color="b", label="Rotated", **arrow_props
            )
            ax.legend()

        for ax in axs.flatten():
            ax.axis("equal")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.grid()
            # ax.legend()

        if savefig:
            figure_name = "rotation_with_perpendicular_basis"
            plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def test_zero_rotation():
    vec_init = np.array([1, 0])
    vec_rot = np.array([1, 0])

    vector_rotation = VectorRotationXd.from_directions(vec_init, vec_rot)

    vec_test = np.array([1, 0])
    vec_rotated = vector_rotation.rotate(vec_test)
    assert np.allclose(vec_test, vec_rotated)

    vec_test = np.array([0, 1])
    vec_rotated = vector_rotation.rotate(vec_test)
    assert np.allclose(vec_test, vec_rotated)


def test_mirror_rotation():
    vec1 = np.array([0, 1])
    vec2 = np.array([1, 0])

    vector_rotation1 = VectorRotationXd.from_directions(vec1, vec2)
    vector_rotation2 = VectorRotationXd.from_directions(vec2, vec1)

    vec_rand = np.ones(2) / np.sqrt(2)

    # Back and forward rotation
    vec_rot = vector_rotation2.rotate(vector_rotation1.rotate(vec_rand))

    assert np.allclose(vec_rand, vec_rot)


def test_cross_rotation_3d():
    vec_init = np.array([1, 0, 0])
    vec_rot = np.array([0, 1, 0])

    vector_rotation = VectorRotationXd.from_directions(vec_init, vec_rot)

    vec_test = np.array([-1, 0, 0])
    vec_rotated = vector_rotation.rotate(vec_test)
    assert np.allclose(vec_rotated, [0, -1, 0]), "Not correct rotation."

    vec_test = np.array([0, 0, 1])
    vec_rotated = vector_rotation.rotate(vec_test)
    assert np.allclose(vec_rotated, vec_test), "No rotation expected."

    vec_test = np.ones(3)
    vec_test = vec_test / LA.norm(vec_test)
    vec_rotated = vector_rotation.rotate(vec_test)
    assert np.isclose(LA.norm(vec_rotated), 1), "Unit norm expected."


def test_multi_rotation_array():
    # Rotation from 1 to final
    vector_seq = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ]
    ).T

    rotation_seq = VectorRotationSequence.create_from_vector_array(vector_seq)
    rotated_vec = rotation_seq.rotate(direction=np.array([1, 0, 0]))
    assert np.allclose(rotated_vec, [0, -1, 0]), "Unexpected rotation."

    rotation_seq = VectorRotationSequence.create_from_vector_array(vector_seq)
    rotated_vec = rotation_seq.rotate_weighted(
        direction=np.array([1, 0, 0]), weights=np.array([0.5, 0.5, 0, 0])
    )
    out_vec = np.array([0, 1, 1]) / np.sqrt(2)
    assert np.allclose(rotated_vec, out_vec), "Not rotated into second plane."


def test_rotation_tree():
    new_tree = VectorRotationTree(root_idx=0, root_direction=np.array([0, 1]))

    new_tree.add_node(node_id=10, direction=np.array([1, 0]), parent_id=0)
    new_tree.add_node(node_id=20, direction=np.array([1, 0]), parent_id=10)

    new_tree.add_node(node_id=30, direction=np.array([-1, 0]), parent_id=0)
    new_tree.add_node(node_id=40, direction=np.array([-1, 0]), parent_id=30)

    # Full rotation in one direction
    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 20, 40],
        weights=[0, 0, 1],
    )

    assert np.isclose(new_tree._graph.nodes[0]["weight"], 1)
    assert np.isclose(new_tree._graph.nodes[10]["weight"], 0)
    assert np.isclose(new_tree._graph.nodes[30]["weight"], 1)
    assert np.isclose(new_tree._graph.nodes[40]["weight"], 1)

    assert np.allclose(new_tree._graph.nodes[40]["direction"], weighted_mean)

    # Equal rotation
    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 20, 40],
        weights=np.ones(3) / 3,
    )
    assert np.isclose(new_tree._graph.nodes[0]["weight"], 1)
    assert np.isclose(new_tree._graph.nodes[10]["weight"], 1.0 / 3)
    assert np.isclose(new_tree._graph.nodes[30]["weight"], 1.0 / 3)

    assert np.allclose(new_tree._graph.nodes[0]["direction"], weighted_mean)

    # Left / right rotation
    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 20, 40],
        weights=[0.0, 0.5, 0.5],
    )
    assert np.isclose(new_tree._graph.nodes[0]["weight"], 1)

    assert np.allclose(new_tree._graph.nodes[0]["direction"], weighted_mean)

    # Left shift (half)
    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 20, 40],
        weights=[0.5, 0.0, 0.5],
    )
    final_dir = (
        new_tree._graph.nodes[0]["direction"] + new_tree._graph.nodes[40]["direction"]
    )
    final_dir = final_dir / LA.norm(final_dir)

    # Check direction
    assert np.allclose(final_dir, weighted_mean)


def test_two_ellipse_with_normal_obstacle():
    # Simple obstacle which looks something like this:
    # ^   <-o
    # |     |
    # o  -  o
    new_tree = VectorRotationTree(root_idx=0, root_direction=np.array([0, 1]))
    new_tree.add_node(node_id=1, direction=np.array([1, 0]), parent_id=0)
    new_tree.add_node(node_id=2, direction=np.array([-0.2, 1]), parent_id=1)
    new_tree.add_node(node_id=3, direction=np.array([-1, 0]), parent_id=2)


def test_multi_normal_tree():
    # Base-normal
    new_tree = VectorRotationTree(root_idx=0, root_direction=np.array([0, 1]))

    # 1st object + normal
    new_tree.add_node(node_id=1, direction=np.array([1, 0]), parent_id=0)
    new_tree.add_node(node_id=2, direction=np.array([0.2, 1.0]), parent_id=1)
    new_tree.add_node(node_id=3, direction=np.array([-1.0, 0.0]), parent_id=2)

    # 2nd object + normal
    new_tree.add_node(node_id=4, direction=np.array([0.0, 1.0]), parent_id=1)
    new_tree.add_node(node_id=5, direction=np.array([-1.0, -0.2]), parent_id=4)
    new_tree.add_node(node_id=6, direction=np.array([0, -1.0]), parent_id=5)

    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 3],
        weights=[0.5, 0.5],
    )

    final_dir = (
        new_tree._graph.nodes[0]["direction"] + new_tree._graph.nodes[3]["direction"]
    )
    final_dir = final_dir / LA.norm(final_dir)
    assert np.allclose(weighted_mean, final_dir)

    # 180 turn and several branches
    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 6],
        weights=[0.5, 0.5],
    )
    assert np.allclose(np.array([-1.0, 0]), weighted_mean)


def test_simple_triple_branch():
    new_tree = VectorRotationTree()

    # Simplified node
    new_tree.set_root(-1, direction=np.array([1, 0]))

    # Add three specific nodes
    new_tree.add_node((0, 0), direction=np.array([1, 0]), parent_id=-1)
    new_tree.add_node((1, 1), direction=np.array([0, 1]), parent_id=-1)
    new_tree.add_node((2, 2), direction=np.array([0, -1]), parent_id=-1)

    node_list = [-1, (0, 0), (1, 1), (2, 2)]
    weights = np.array([0.2128574, 0.02090366, 0.74444571, 0.02179324])

    averaged_dir = new_tree.get_weighted_mean(node_list=node_list, weights=weights)

    # Main weight on up-direction -> go right-up
    assert averaged_dir[0] > 0 and averaged_dir[1] > 0


def test_double_branch_tree():
    # Simple-double-part tree
    direction_tree = VectorRotationTree()
    direction_tree.set_root(
        root_idx=-1,
        direction=np.array([1.0, 0]),
    )
    direction_tree.add_node(
        node_id=0,
        parent_id=-1,
        direction=np.array([1.0, 0.0]),
    )

    direction_tree.add_node(
        node_id=1,
        parent_id=-1,
        direction=np.array([0.0, 1.0]),
    )

    main_direction = np.array([-1.0, 0.0])
    direction_tree.add_node(node_id=2, parent_id=1, direction=main_direction)
    weight = 1.0
    averaged_direction = direction_tree.get_weighted_mean(
        node_list=[0, 2], weights=[(1 - weight), weight]
    )

    assert (
        np.dot(main_direction, averaged_direction) > 0.99
    ), "Expected to be close to large weight"


def test_tree_assembly_and_reduction():
    sqrt2 = np.sqrt(2) / 2
    sequence1 = VectorRotationSequence.create_from_vector_array(
        np.array([[1.0, 0.0], [sqrt2, sqrt2], [0.0, 1.0]]).T
    )
    sequence2 = VectorRotationSequence.create_from_vector_array(
        np.array([[1.0, 0.0], [sqrt2, -sqrt2], [0.0, -1.0]]).T
    )

    new_tree = VectorRotationTree.from_sequence(
        sequence=sequence1, root_id=0, node_id=1
    )
    new_tree.add_sequence(sequence=sequence2, parent_id=0, node_id=2)
    merged_sequence = new_tree.reduce_weighted_to_sequence(
        node_list=[1, 2], weights=[0.5, 0.5]
    )
    weighted_vector = merged_sequence.get_end_vector()
    assert np.allclose(weighted_vector, [1.0, 0.0]), "Unexpected weighting."
    assert merged_sequence.n_rotations == 2, "Incorrect rotation-level after reduction."


def test_tree_assembly_and_reduction_3d():
    sqrt2 = np.sqrt(2) / 2
    sequence1 = VectorRotationSequence.create_from_vector_array(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, sqrt2, sqrt2]]).T
    )
    sequence2 = VectorRotationSequence.create_from_vector_array(
        np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -sqrt2, -sqrt2]]).T
    )

    new_tree = VectorRotationTree.from_sequence(
        sequence=sequence1, root_id=0, node_id=1
    )
    new_tree.add_sequence(sequence=sequence2, parent_id=0, node_id=2)
    merged_sequence = new_tree.reduce_weighted_to_sequence(
        node_list=[1, 2], weights=[0.5, 0.5]
    )
    weighted_vector = merged_sequence.get_end_vector()
    assert np.allclose(weighted_vector, [1.0, 0.0, 0.0]), "Unexpected weighting."
    assert merged_sequence.n_rotations == 2, "Incorrect rotation-level after reduction."


def test_rotation_of_sequence():
    sqrt2 = np.sqrt(2) / 2
    sequence = VectorRotationSequence.create_from_vector_array(
        np.array([[1.0, 0.0], [sqrt2, sqrt2], [0.0, 1.0]]).T
    )
    rotation = VectorRotationXd.from_directions([0.0, -1.0], [-1, -1])

    rotated_sequence = rotation.rotate_sequence(sequence)
    assert np.allclose(rotated_sequence.get_end_vector(), [sqrt2, sqrt2])

    vector = np.array([2, 3])
    vector = vector / np.linalg.norm(vector)
    rot_vect1 = sequence.rotate(vector)
    rot_vect2 = rotated_sequence.rotate(vector)
    assert np.allclose(
        rot_vect1, rot_vect2
    ), "Rotation should remain for vectors in the plane."


def test_parallel_vector_of_different_magnitude():
    """Check that it works with even vectors."""
    v1 = np.array([4.0, 0.0])
    v2 = np.array([1.0, 0.0])

    rotation = VectorRotationXd.from_directions(v1, v2)
    assert not np.any(np.isnan(rotation.base))

    sequence = VectorRotationSequence.create_from_vector_array(np.vstack((v1, v2)).T)
    assert not np.any(np.isnan(sequence.basis_array))


def test_perpendicular_sequence():
    v1 = np.array([3.0, -3.0])
    v2 = np.array([0.70710678, -0.70710678])
    sequence = VectorRotationSequence.create_from_vector_array(np.vstack((v1, v2)).T)
    assert np.isclose(sequence.rotation_angles[0], 0)


def test_graph_reduction():
    new_tree = VectorRotationTree()

    vec_nominal = np.array([-1.0, 0])

    new_tree.set_root(0, vec_nominal)
    new_tree.add_node(node_id=1, parent_id=0, direction=vec_nominal)
    new_tree.add_node(node_id=2, parent_id=1, direction=vec_nominal)
    new_tree.add_node(node_id=3, parent_id=2, direction=vec_nominal)

    # First branch
    new_tree.add_node(node_id=4, parent_id=3, direction=vec_nominal)

    # Second branch
    vec1 = np.array([-0.941709, -0.336428])
    new_tree.add_node(node_id=(5, 0), parent_id=3, direction=vec1)
    vec2 = np.array([1.0, 0.0])
    new_tree.add_node(node_id=(5, 1), parent_id=(5, 0), direction=vec2)

    angles1 = math.acos(vec_nominal @ vec1)
    angles2 = math.acos(vec1 @ vec2)

    node_list = [4, (5, 1), 1]
    weights = [0.00, 1.0, 0.0]

    sequence = new_tree.reduce_weighted_to_sequence(node_list, weights)
    assert math.isclose(angles1, sequence.rotation_angles[0], abs_tol=1e-6)
    assert math.isclose(angles2, sequence.rotation_angles[1], abs_tol=1e-6)

    averaged_direction = sequence.get_end_vector()
    assert (
        abs(averaged_direction[1] / averaged_direction[0]) < 1e-1
    ), "Strongly align with last vector"


def test_rotate_direction():
    base = np.array([[-1, 0.0], [0, -1.0]])
    rotation_angle = math.pi * 0.5

    rotation = VectorRotationXd(base, rotation_angle)
    final_vector = rotate_direction(base[:, 0], base, rotation_angle)
    assert np.allclose(final_vector, [0, -1])


def test_create_overrotation():
    vec1 = np.array([-0.941709, -0.336428])
    vec2 = np.array([1.0, 0.0])

    vector_rotation = VectorRotationXd.from_directions(vec1, vec2)
    assert vector_rotation.rotation_angle > math.pi * 0.5
    assert np.allclose(vec2, vector_rotation.rotate(vec1))


def test_rotation_from_parallel_vectors():
    # Highly similar vectors 3D
    vec1 = np.array([-0.20744130246674772, 0.9767642733625228, 0.05384849406884028])
    vec2 = np.array([-0.20744130246674777, 0.976764273362523, 0.05384849406884029])
    rotation = VectorRotationXd.from_directions(vec1, vec2)
    assert not np.any(np.isnan(rotation.base))
    assert math.isclose(rotation.rotation_angle, 0, abs_tol=1e-6)

    # Identical vectors 2D
    vec = np.array([0, 1])
    rotation = VectorRotationXd.from_directions(vec, vec)
    assert not np.any(np.isnan(rotation.base))
    assert math.isclose(rotation.rotation_angle, 0)


def test_sequence_of_similar_vectors():
    vec1 = np.array([0.0, -0.11245940024922693, 0.06168989053606522])
    vec2 = np.array([0.0, 0.2, -0.10999999999999999])
    rotation = VectorRotationXd.from_directions(vec1, vec2)
    assert np.allclose(
        rotation.get_second_vector(), vec2 / np.linalg.norm(vec2)
    ), "Restoring worked."

    vec1 = np.array([0.06376571583361212, -0.043198437634528145, -0.003709748681816061])
    vec2 = np.array([0.06376571583361212, -0.04319843763452815, -0.003709748681816061])

    rotation = VectorRotationSequence.create_from_vector_array(
        np.vstack((vec1, vec2)).T
    )
    assert not np.any(np.isnan(rotation.basis_array))
    assert np.allclose(rotation.rotation_angles, [0])


def test_end_of_sequence():
    base = np.array([-1.0, 0.0])
    final = np.array([0.655, 0.756])
    final = final / np.linalg.norm(final)
    rotation = VectorRotationSequence.create_from_vector_array(
        np.vstack((base, final)).T
    )

    final_reconstruct = rotation.get_end_vector()

    assert np.isclose(np.cos(rotation.rotation_angles[0]), np.dot(base, final))
    assert np.allclose(final, final_reconstruct)


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    test_end_of_sequence()
    test_sequence_of_similar_vectors()

    test_graph_reduction()

    test_rotation_from_parallel_vectors()

    test_create_overrotation()
    test_rotate_direction()

    test_perpendicular_sequence()

    test_parallel_vector_of_different_magnitude()

    test_rotation_of_sequence()

    test_tree_assembly_and_reduction()
    test_tree_assembly_and_reduction_3d()

    test_double_branch_tree()

    test_cross_rotation_2d(visualize=False, savefig=0)
    test_zero_rotation()
    test_cross_rotation_3d()
    test_multi_rotation_array()

    test_rotation_tree()
    test_multi_normal_tree()
    test_simple_triple_branch()

    print("\nDone with tests.")
