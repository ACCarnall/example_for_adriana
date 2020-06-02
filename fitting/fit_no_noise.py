import numpy as np
import bagpipes as pipes
import matplotlib.pyplot as plt
from astropy.io import fits

import sys
sys.path.append("../utils")
from load_vandels import *
from cat_filt_list import *


def get_fit_instructions():

    dblplaw = {}
    dblplaw["alpha"] = (0.1, 1000.)
    dblplaw["alpha_prior"] = "log_10"
    dblplaw["beta"] = (0.1, 1000.)
    dblplaw["beta_prior"] = "log_10"
    dblplaw["tau"] = (0.1, 5.)
    dblplaw["massformed"] = (1., 13.)
    dblplaw["massformed_prior"] = "log_10"
    dblplaw["metallicity"] = (0.01, 2.5)
    dblplaw["metallicity_prior"] = "log_10"

    dust = {}
    dust["type"] = "Salim"
    dust["eta"] = 2.
    dust["Av"] = (0., 3.)
    dust["delta"] = (-0.3, 0.3)
    dust["delta_prior"] = "Gaussian"
    dust["delta_prior_mu"] = 0.
    dust["delta_prior_sigma"] = 0.1
    dust["B"] = (0., 5.)

    nebular = {}
    nebular["logU"] = -3.

    calib = {}
    calib["type"] = "polynomial_bayesian"
    calib["0"] = (0.5, 1.5) # Zero order is centred on 1, at which point there is no change to the spectrum.
    calib["0_prior"] = "Gaussian"
    calib["0_prior_mu"] = 1.0
    calib["0_prior_sigma"] = 0.25
    calib["1"] = (-0.5, 0.5) # Subsequent orders are centred on zero.
    calib["1_prior"] = "Gaussian"
    calib["1_prior_mu"] = 0.
    calib["1_prior_sigma"] = 0.25
    calib["2"] = (-0.5, 0.5)
    calib["2_prior"] = "Gaussian"
    calib["2_prior_mu"] = 0.
    calib["2_prior_sigma"] = 0.25

    fit_instructions = {}
    fit_instructions["redshift"] = 0.
    fit_instructions["veldisp"] = (100.,400.)
    fit_instructions["veldisp_prior"] = "log_10"
    fit_instructions["t_bc"] = 0.01
    fit_instructions["dblplaw"] = dblplaw
    fit_instructions["dust"] = dust
    fit_instructions["nebular"] = nebular
    fit_instructions["calib"] = calib

    return fit_instructions


# Load list of objects to be fitted from catalogue.
IDs = np.array(["UDS-HST020821SELECT"])
redshifts = np.array([1.0909])

# Fit the catalogue of objects.
cat_filt_list = get_cat_filt_list(IDs)
fit_instructions = get_fit_instructions()
fit_cat = pipes.fit_catalogue(IDs, fit_instructions, load_vandels, run="no_noise",
                              cat_filt_list=cat_filt_list, vary_filt_list=True,
                              redshifts=redshifts, redshift_sigma=0.005,
                              make_plots=True, time_calls=True,
                              full_catalogue=True, n_posterior=1000)

fit_cat.fit(verbose=False)
