""" Custom bayesian mixutre model """
# Author: Lukas Huber <hubernikus@gmail.com>
# License: MIT

import math
import numpy as np

# from scipy.special import betaln, digamma, gammaln

# from sklearn.mixture import BaseMixture
# from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture


class GaussianMixtureContinous(BayesianGaussianMixture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _initialize(self, X, resp):
        pass

    def _estimate_weights(self, nk):
        pass

    def _estimate_means(self, nk, xk):
        pass

    def _estimate_precisions(self, nk, xk, sk):
        pass
