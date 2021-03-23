import numpy as np
import numba as nb

from iminuit import Minuit
from iminuit.util import describe, make_func_code

############################################################
kwd = {'parallel': False, 'fastmath': True}

@nb.njit(**kwd)
def custom_chi2(y, y_err, ym):
    return np.sum(np.where(y > 0, (y-ym)**2/y_err**2, 0))

@nb.njit(**kwd)
def custom_errors(y):
    return np.where(y > 0, np.sqrt(np.abs(y)), 1)

############################################################
@nb.njit(**kwd)
def norm_pdf(x, mu, sigma):
    invs = 1.0 / sigma
    z = (x - mu) * invs
    invnorm = 1 / np.sqrt(2 * np.pi) * invs
    return np.exp(-0.5 * z ** 2) * invnorm

@nb.njit(**kwd)
def uniform_pdf(x):
    obs_left, obs_right = 230, 350 # this is badly hardcoding, i know
    x_in_range = (x >= obs_left) & (x <= obs_right)
    return np.where(x_in_range, 1/(obs_right - obs_left), 0)

@nb.njit(**kwd)
def model(x, mu, sigma, fr, N):
    return N*(fr*norm_pdf(x, mu, sigma) + (1-fr)*uniform_pdf(x))

############################################################
class MyChi2:
    def __init__(self, model, x, y):
        self.model = model  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        self.y_err = custom_errors(self.y)
        self.func_code = make_func_code(describe(self.model)[1:])

    def __call__(self, *par):  # we accept a variable number of model parameters
        ym = self.model(self.x, *par)
        chi2 = custom_chi2(self.y, self.y_err, ym)
        return chi2
