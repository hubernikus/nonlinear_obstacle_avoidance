#!/usr/bin/env python

import os
import logging
from dataclasses import dataclass

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from scipy.io import loadmat


class HandwrittingDataHandler:
    def __init__(self, dataset_name, dataset_dir=None):
        if dataset_dir is None:
            dataset_dir = "/home/lukas/Code/handwritting_dataset/DataSet"

        file_name = os.path.join(dataset_dir, dataset_name)
        self.data = loadmat(file_name)
        logging.info("Finished data loading.")

    @property
    def dimensions(self):
        return self.data["demos"][0][0][0][0][0].shape[0]

    @property
    def dt(self):
        return self.data["dt"][0][0]

    @property
    def n_demonstrations(self):
        return self.data["demos"][0].shape[0]

    def get_demonstration(self, it_demo):
        return self.data["demos"][0][it_demo][0][0]

    def get_positions(self, it_demo):
        # demo[0][0]['pos']
        return self.data["demos"][0][it_demo][0][0][0]

    def get_times(self, it_demo):
        # demo[0][0]['vel']
        return self.data["demos"][0][it_demo][0][0][1]

    def get_velocities(self, it_demo):
        return self.data["demos"][0][it_demo][0][0][2]

    def get_accelerations(self, it_demo):
        return self.data["demos"][0][it_demo][0][0][3]

    def get_dt(self, it_demo=0):
        """Returns the delta-time for a specific demo.
        A default argument is given as we assume to have the same dt for all demos."""
        return self.data["demos"][0][it_demo][0][0][4][0][0]


@dataclass(slots=True)
class DirectionSpaceKernel:
    null_matrix: np.ndarray
    direction: np.ndarray


class StarshapedLyapunov:
    """Inspired by GMM see
    https://moodle.epfl.ch/pluginfile.php/2851387/mod_resource/content/2/assignment.html
    """

    def __init__(self, data_handler, n_gmms: int = 5):
        positions = data_handler.get_positions(0)
        velocities = data_handler.get_velocities(0)

        # Only take points with nonzero-velocities
        weight_vel = LA.norm(velocities, axis=0)
        ind_nonzero = weight_vel > 0

        if not np.sum(ind_nonzero):
            raise ValueError("Only got zero valued velocities")

        self.positions = positions[:, ind_nonzero]
        self.velocities = velocities[:, ind_nonzero]
        self.weight_vel = weight_vel[:, ind_nonzero]

        self.dimensions = data_handler.dimensions()

        self.priors = np.array(n_gmms)
        self.mean_positions = np.array(np.array(n_gmms))

    def regress(self, it_max: int = 10):
        self.initialization_step()

        for ii in range(it_max):
            self.expectation_step()
            self.maximazation_step()
            pass

    def initialization_step(self):
        """Initialize priors α={α1,…,αK}, means μ={μ1,…,μK} and
        Covariance matrices Σ={Σ1,…,ΣK}
        """
        pass

    def expectation_step(self):
        """Expectation Step: For each Gaussian k∈{1,…,K},
        compute the probability that it is responsible for each point xi
        in the dataset."""
        pass

    def maximazation_step(self):
        """Maximization Step: Re-estimate the priors α={α1,…,αK},
        means μ={μ1,…,μK} and Covariance matrices Σ={Σ1,…,ΣK}"""
        pass

    def get_log_likelihood(self):
        pass

    def directionspace_kernel(self):
        pass


if (__name__) == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    my_handler = HandwrittingDataHandler(
        dataset_name="Angle.mat",
        dataset_dir="/home/lukas/Code/handwritting_dataset/DataSet",
    )

    my_starshape = StarshapedLyapunov(my_handler)

    logging.info("Finished running.")
