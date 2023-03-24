"""
The learning.graph includes elements to automatically build graphs of GMM's to
simplify the model creation
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import get_angle_space_of_array
from vartools.directional_space import get_angle_space_inverse_of_array

from vartools.dynamical_systems import LinearSystem, LocallyRotated

# from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system

from dynamic_obstacle_avoidance.obstacles import Ellipse
from nonlinear_avoidance.multiboundary_container import (
    MultiBoundaryContainer,
)
from nonlinear_avoidance.rotation_container import RotationContainer

from nonlinear_avoidance.gmm_learner.learner.directional_gmm import DirectionalGMM
from nonlinear_avoidance.gmm_learner.visualization.gmm_visualization import (
    draw_gaussians,
)
from nonlinear_avoidance.gmm_learner.learner.visualizer import (
    plot_graph_and_gaussians,
    plot_obstacle_wall_environment,
)


# TODO: fix mess of 'container in container' (i.e. learned GMM & graph)
# make one out of it.


# class GraphGMM(RotationContainer):
class GraphGMM(MultiBoundaryContainer):
    """Creats grpah from input gmm

    The main idea is (somehow to an overcompetivtive faimliy):
    - Direct-successor are friends (Grand-grand-...-parents to grand-grand-...-child)
    - Sibilings (plus cousins and all not successors) are rivals

    This is additionally an obstacle container (since graph) with multiple hulls.
    No inheritance from BaseContainer due to desired difference in behavior as
    this class has a graph-like structure.
    """

    # TODO: Currently this is a child of MultiBoundaryContainer
    # -> the coupling / inheritance should be reduced for readability and adaptability...

    def __init__(
        self, file_name, n_gaussian, LearnerType=DirectionalGMM, *args, **kwargs
    ):
        # Ellipse factor relative to
        self._ellipse_axes_factor = 1

        self._obstacle_list = []

        # End point is the the direction of intersection
        self._end_points = None
        self._graph_root = None

        self._Learner = DirectionalGMM()
        self._Learner.load_data_from_mat(file_name=file_name)
        self._Learner.regress(n_gaussian=n_gaussian)

        # TODO: this should already be created in "rotation_container"
        # self._ConvergenceDynamics = [None for ii in range(len(self))]
        self._ConvergenceDynamics = [None for ii in range(len(self))]

        super().__init__()

    @property
    def gmm(self):
        return self._Learner.dpgmm

    @property
    def n_gaussians(self):
        # Depreciated -> use 'n_components' instead
        return self._Learner.dpgmm.covariances_.shape[0]

    @property
    def n_components(self):
        return self._Learner.dpgmm.covariances_.shape[0]

    @property
    def dim_space(self):
        return self._Learner.dim_space

    @property
    def null_ds(self):
        # DS going to the attractor
        return self._Learner.null_ds

    @property
    def pos_attractor(self):
        # DS going to the attractor
        return self._Learner.pos_attractor

    @property
    def attractor_position(self):
        # DS going to the attractor
        return self._Learner.pos_attractor

    @property
    def _attractor_position(self):
        # DS going to the attractor
        return self._Learner.pos_attractor

    @_attractor_position.setter
    def _attractor_position(self, value):
        # DS going to the attractor
        self._Learner.pos_attractor = value

    @property
    def convergence_attractor(self) -> bool:
        return self._Learner.convergence_attractor

    @convergence_attractor.setter
    def convergence_attractor(self, value: bool) -> None:
        self._Learner.convergence_attractor = value

    @property
    def _ConvergenceDynamics(self):
        return self._convergence_dynamics

    @_ConvergenceDynamics.setter
    def _ConvergenceDynamics(self, value):
        self._convergence_dynamics = value

    @property
    def positions(self):
        return self._Learner.pos

    def evaluate(self, position):
        return self.predict(position)

    def predict(self, position):
        return self._Learner.predict(position)

    # def get_convergence_direction(self, position, it_obs):
    #     """ Get the (null) direction for a specific gaussian-hull in the multi-body-boundary
    #     container which serves for the rotational-modulation.

    #     The direction is based on a locally linear dynamical-system. """
    #     # Check if attractor is in current object
    #     attr= self._get_local_attractor(it_obs)
    #     # return  evaluate_linear_dynamical_system(position=position, center_position=attr)
    #     return LinearSystem(attractor_position=attr).evaluate

    def _get_local_attractor(self, it_obs: int) -> Optional[np.ndarray]:
        """Returns local_attractor based projected point & parent_direction."""
        # TODO: maybe parent _end_points / end-point direction could be used...
        if (
            self[it_obs].get_gamma(
                self.pos_attractor, in_global_frame=True, relative_gamma=False
            )
            >= 1
        ):
            local_attractor = self.pos_attractor
        else:
            it_parent = self.get_parent(it_obs)
            # rel_dir =  self[it_parent].center_position - self[it_obs].center_position
            rel_dir = self[it_obs].center_position - self[it_parent].center_position

            # Otherwise use the 'connection point' [which was chosen as global connection]
            local_attractor = self[it_obs].get_intersection_with_surface(
                edge_point=self._end_points[:, it_obs],
                direction=(-1) * rel_dir,
                in_global_frame=True,
            )
        if local_attractor is None:
            breakpoint()

        return local_attractor

    def get_xy_lim_plot(self):
        """Return (x_lim, y_lim) tuple based on recorded dataset."""
        return self._Learner.get_xy_lim_plot()

    def get_mixing_weights(self, *args, **kwargs):
        return self._Learner.get_mixing_weights(*args, **kwargs)

    def ellipse_axes_length(self, it, axes_factor=3):
        """Get axes length of ellipses extracted from the GMM-covariances."""
        # Get intersection with circle and then ellipse hull
        if self.gmm.covariance_type == "full":
            covariances = self.gmm.covariances_[it, :, :][: self.dim_space, :][
                :, : self.dim_space
            ]
        else:
            raise TypeError("Not implemented for unfull covariances")
        covariances = covariances * axes_factor**self.dim_space
        return covariances

    def get_end_point(self, it):
        """Return the end point of a specific gaussian parameter.

        Note: this is (simplified) since we assume the orienation being the
        same as it is at the center."""

        mean = self.gmm.means_[it, :]

        mean_pos = mean[: self.dim_space]
        mean_dir = mean[-(self.dim_space - 1) :]

        null_direction = self.null_ds.evaluate(mean_pos)

        center_velocity_dir = get_angle_space_inverse(
            dir_angle_space=mean_dir, null_direction=null_direction
        )

        if self.dim_space > 2:
            raise NotImplementedError()
        # 2D only (!) -- temporary; exand this!
        covariances = self.gmm.covariances_[it][: self.dim_space, :][
            :, : self.dim_space
        ]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        # angle = 180 * angle / np.pi      # Convert to degrees
        # v = 2. * np.sqrt(2.) * np.sqrt(v)
        v = np.sqrt(2.0) * np.sqrt(v)

        angle = -angle
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )

        axes_lengths = self.ellipse_axes_length(
            it=it, axes_factor=self._ellipse_axes_factor
        )
        end_point = mean_pos + axes_lengths.dot(center_velocity_dir)

        center_velocity_dir = rot_mat.T @ center_velocity_dir
        fac = np.sum(center_velocity_dir**2 / v**2)
        center_velocity_dir = center_velocity_dir / np.sqrt(fac)
        # center_velocity_dir = center_velocity_dir * v
        center_velocity_dir = rot_mat @ center_velocity_dir

        # End point somewhere in cluster-intersection region
        end_point = mean_pos + center_velocity_dir

        return end_point

    def get_rival_weight(
        self,
    ):
        """ """
        end_point = np.arange(3)
        pass

    def _get_assigned_elements(self):
        """Get the list of all assigned elements.
        This is useful only during of the graph."""
        if self._graph_root is None:
            return []
        graph = [self._graph_root]
        for sub_list in self._children_list:
            graph = graph + sub_list
        return graph

    @property
    def number_of_levels(self):
        raise NotImplementedError()
        # return -1

    @property
    def number_of_branches(self):
        """Returns the number of brances by counting the number of 'dead-ends'."""
        return np.sum([not (len(list)) for list in self._children_list])

    def create_graph_from_gaussians(self):
        """Create graph from learned GMM."""
        self._end_points = np.zeros((self.dim_space, self.n_gaussians))
        self._parent_array = (-1) * np.ones(self.n_gaussians, dtype=int)
        self._children_list = [[] for ii in range(self.n_gaussians)]

        # First 'level' is based on closest to origin
        for ii in range(self.n_gaussians):
            self._end_points[:, ii] = self.get_end_point(it=ii)

        # Get root / main-parent (this could be replaced by optimization / root finding)
        dist_atrractors = np.linalg.norm(
            self._end_points - np.tile(self.pos_attractor, (self.n_gaussians, 1)).T,
            axis=0,
        )

        self._graph_root = np.argmin(dist_atrractors)

        weights = self.get_mixing_weights(
            X=self._end_points.T,
            weight_factor=self.dim_space,
            feat_in=np.arange(self.dim_space),
            feat_out=-np.arange(self.dim_space, 1),
        ).T

        weights_without_self = np.copy(weights)
        for gg in range(self.n_gaussians):
            weights_without_self[gg, gg] = -1

        parents_preference = np.argsort(weights_without_self, axis=1)
        parents_preference = np.flip(parents_preference, axis=1)

        it_count = 0
        while True:
            it_count += 1
            if it_count > 100:
                # TODO: remove it_count (only for debugging purposes)
                breakpoint()
            ind_assigned = self._get_assigned_elements()
            if len(ind_assigned) == self.n_gaussians:
                break
            ind_unassigned = [
                ii for ii in range(self.n_gaussians) if ii not in ind_assigned
            ]

            print("ind_assigned", ind_assigned)

            for it_pref in range(self.n_gaussians - 1):
                list_parent_child = []
                for ind_child in ind_unassigned:
                    # TODO: to speed up only look at last index (new parents),
                    # when the last run fully resolved
                    if parents_preference[ind_child, it_pref] in ind_assigned:
                        list_parent_child.append(
                            (parents_preference[ind_child, it_pref], ind_child)
                        )

                print(f"List at {ii} is {list_parent_child}")
                if len(list_parent_child):
                    # If it_pref > 0 it means that the prefered parent choice would be another one
                    # in that case the prefered graph might be a different one (!)
                    # Hence it will be double checked that this is actually the optimal one!

                    if it_pref and len(list_parent_child) > 1:
                        # If the chosen one is the secondary choice, only adapt one
                        # and then reitarate, i.e. make an optimal sequence / graph
                        ind_critical = [child for par, child in list_parent_child]
                        ind_child = np.argmax(
                            weights_without_self[ind_critical, it_pref]
                        )
                        ind_child = ind_critical[ind_child]

                        self.extend_graph(
                            parent=parents_preference[ind_child, it_pref],
                            child=ind_child,
                        )
                        # TODO: Extesnively test this... Make sure it's working correctly.
                    else:
                        # Assign all elements if it's the first choice
                        for ind_parent, ind_child in list_parent_child:
                            self.extend_graph(child=ind_child, parent=ind_parent)
                    break

    def set_convergence_directions(self, NonlinearDynamcis=None):
        # TODO: Get rotation @ center & use this as direction.
        attractor = NonlinearDynamcis.attractor_position

        for it_obs in range(self.n_obstacles):
            local_attractor = self._get_local_attractor(it_obs=it_obs)
            local_velocity = local_attractor - self[it_obs].center_position

            ds_direction = get_angle_space(
                direction=local_velocity,
                null_direction=(attractor - self[it_obs].center_position),
            )

            reference_radius = self[it_obs].get_reference_length()
            # self._ConvergenceDynamics[it_obs] = LocallyRotated(
            # mean_rotation=ds_direction, rotation_center=self[it_obs].center_position,
            # influence_radius=reference_radius, attractor_position=attractor)

            if it_obs < len(self._ConvergenceDynamics):
                self._ConvergenceDynamics[it_obs] = LinearSystem(
                    attractor_position=local_attractor
                )
            else:
                self._ConvergenceDynamics.append(
                    LinearSystem(attractor_position=local_attractor)
                )

    def create_learned_boundary(self, oversize_factor=2.0):
        """Adapt (inflate) each gaussian-ellipse to the graph such that there is a overlap.
        The simple gaussians are transformed to 'Obstacles'"""

        if self.dim_space != 2:
            # 2D only (!) -- temporary; exand this!
            raise NotImplementedError()

        for gg in range(self.n_gaussians):
            covariances = self.gmm.covariances_[gg][: self.dim_space, :][
                :, : self.dim_space
            ]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            v = np.sqrt(2.0) * np.sqrt(v)

            # Don't consider them as boundaries (yet)
            # self._obstacle_list.append(
            self.append(
                Ellipse(
                    center_position=self.gmm.means_[gg, : self.dim_space],
                    orientation=angle,
                    axes_length=v * oversize_factor,
                    is_boundary=True,
                )
            )

        # prop_dist_end = np.zeros(self.n_gaussians)
        # for ii in range(self.n_gaussians):
        # it_parent = self.get_parent(ii)
        # prop_dist_end[ii] = self._obstacle_list[it_parent].get_gamma(
        # self._end_points[:, ii], in_global_frame=True)
        # pass

        self.create_graph_from_gaussians()

    def plot_obstacle_wall_environment(self, **kwargs):
        """Plot the environment such that we have an 'inverse' obstacle avoidance."""
        plot_obstacle_wall_environment(self, **kwargs)

    def plot_graph_and_gaussians(self, **kwargs):
        """Plot the graph and the gaussians as 'grid'."""
        plot_graph_and_gaussians(self, **kwargs)

    def eval_weights(self):
        pass
