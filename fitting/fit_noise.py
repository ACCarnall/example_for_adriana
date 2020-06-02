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

    noise = {}
    noise["type"] = "GP_exp_squared"
    noise["scaling"] = (0.1, 10.)
    noise["scaling_prior"] = "log_10"
    noise["norm"] = (0.0001, 1.)
    noise["norm_prior"] = "log_10"
    noise["length"] = (0.01, 1.)
    noise["length_prior"] = "log_10"

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
    fit_instructions["noise"] = noise
    fit_instructions["calib"] = calib

    return fit_instructions


def analysis_func(fit):
    import matplotlib.pyplot as plt
    fit.posterior.get_advanced_quantities()

    fig = plt.figure(figsize=(12, 5.))
    ax = plt.subplot()

    y_scale = pipes.plotting.add_spectrum(fit.galaxy.spectrum, ax)
    pipes.plotting.add_spectrum_posterior(fit, ax, y_scale=y_scale)

    noise_post = fit.posterior.samples["noise"]*10**-y_scale
    noise_perc = np.percentile(noise_post, (16, 50, 84), axis=0).T
    noise_max = np.max(np.abs(noise_perc))
    noise_perc -= 1.05*noise_max

    ax.plot(fit.galaxy.spectrum[:,0], noise_perc[:, 1], color="darkorange")

    ax.fill_between(fit.galaxy.spectrum[:,0], noise_perc[:, 0],
                    noise_perc[:, 2], color="navajowhite", alpha=0.7)

    ymax = ax.get_ylim()[1]
    ax.set_ylim(-2.1*noise_max, ymax)
    ax.axhline(0., color="gray", zorder=1, lw=1.)
    ax.axhline(-1.05*noise_max, color="gray", zorder=1, lw=1., ls="--")

    plt.savefig("pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_gp.pdf",
                bbox_inches="tight")

    plt.close()


# Load list of objects to be fitted from catalogue.
IDs = np.array(["UDS-HST020821SELECT"])
redshifts = np.array([1.0909])

# Fit the catalogue of objects.
cat_filt_list = get_cat_filt_list(IDs)
fit_instructions = get_fit_instructions()
fit_cat = pipes.fit_catalogue(IDs, fit_instructions, load_vandels, run="noise",
                              cat_filt_list=cat_filt_list, vary_filt_list=True,
                              redshifts=redshifts, redshift_sigma=0.005,
                              make_plots=True, time_calls=True,
                              full_catalogue=True, n_posterior=1000,
                              analysis_function=analysis_func)

fit_cat.fit(verbose=False)
