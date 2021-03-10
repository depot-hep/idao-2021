import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import zfit
import mplhep
mplhep.set_style('CMS')

def counts_to_datapoints(ima_x, ima_y, obs_left, obs_right):
    obs_edges = np.array(range(obs_left, obs_right + 1))
    obs_bin_centers = (obs_edges[:-1] + obs_edges[1:])/2

    ### NB: this is exact data
    x_samples = np.repeat(obs_bin_centers, ima_x.astype(int))
    y_samples = np.repeat(obs_bin_centers, ima_y.astype(int))

    ### NB: but this is sampling from a hist distribution (aka multinomial): can be used to reduce the number of datapoints
    # take_fraction = .2
    # x_samples = np.random.choice(obs_bin_centers, size=int(take_fraction*ima_x.sum()), p=ima_x/ima_x.sum())
    # y_samples = np.random.choice(obs_bin_centers, size=int(take_fraction*ima_y.sum()), p=ima_y/ima_y.sum())
    return x_samples, y_samples

def build_model(obs, obs_left, obs_right, init_values):
    # signal component
    mu = zfit.Parameter("mu", init_values['mu'], obs_left, obs_right)
    sigma = zfit.Parameter("sigma", init_values['sigma'], 0.01, obs_right - obs_left)
    signal = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs, name='sig')

    # background component
    background = zfit.pdf.Uniform(obs_left, obs_right, obs=obs, name='bkgr')
    # lambd = zfit.Parameter("lambda", -0.01, -1, -0.000001)
    # background = zfit.pdf.Exponential(lambd, obs=obs)

    # combing sig and bkgr together
    fr = zfit.Parameter("fr", init_values['fr'], 0, 1)
    model = zfit.pdf.SumPDF([signal, background], fracs=fr)
    # n_bkg = zfit.Parameter('n_bkg', sum(ima_x))
    # n_sig = zfit.Parameter('n_sig', 1000)
    # gauss_extended = gauss.create_extended(n_sig)
    # exp_extended = exponential.create_extended(n_bkg)
    # uni_extended = uniform.create_extended(n_bkg)
    # model = zfit.pdf.SumPDF([gauss_extended, uni_extended])
    return model

def plot_projections(obs_grid, n_bins, data_proj_dict, model_proj_dict, plot_scaling_dict, fit_params_dict, savefig=True, image_name=None):
    fig, axs = plt.subplots(1, 2, figsize=(20,7))
    data_projx = data_proj_dict['x']
    data_projy = data_proj_dict['y']
    #
    model_projx = model_proj_dict['x']['model']
    model_projx_sig = model_proj_dict['x']['sig']
    model_projx_bkgr = model_proj_dict['x']['bkgr']
    #
    model_projy = model_proj_dict['y']['model']
    model_projy_sig = model_proj_dict['y']['sig']
    model_projy_bkgr = model_proj_dict['y']['bkgr']
    #
    plot_scaling_x = plot_scaling_dict['x']
    plot_scaling_y = plot_scaling_dict['y']
    ##
    axs[0].plot(obs_grid, model_projx*plot_scaling_x, label="Sum - Model")
    # axs[0].plot(obs_grid, model_projx_sig*plot_scaling_x, label="Gauss - Signal")
    # axs[0].plot(obs_grid, model_projx_bkgr*plot_scaling_x, label="Background")
    mplhep.histplot(np.histogram(data_projx, bins=n_bins), yerr=True, color='black', histtype='errorbar',
                    markersize=17, capsize=2.5,
                    markeredgewidth=1.5, zorder=1,
                    elinewidth=1.5, ax=axs[0]
                    )
    axs[0].set_title('X projection')
    if fit_params_dict:
        assert 'x' in fit_params_dict
        assert 'mu' in fit_params_dict['x'] and 'sigma' in fit_params_dict['x'] and 'fr' in fit_params_dict['x']
        mu_patch = mpatches.Patch(color='none', label=f"mu = {fit_params_dict['x']['mu']:.2f}")
        sigma_patch = mpatches.Patch(color='none', label=f"sigma = {fit_params_dict['x']['sigma']:.2f}")
        fr_patch = mpatches.Patch(color='none', label=f"fr = {fit_params_dict['x']['fr']:.4f}")
        axs[0].legend(handles=[mu_patch, sigma_patch, fr_patch])
    ##
    axs[1].plot(obs_grid, model_projy*plot_scaling_y, label="Sum - Model")
    # axs[1].plot(obs_grid, model_projy_sig*plot_scaling_y, label="Gauss - Signal")
    # axs[1].plot(obs_grid, model_projy_bkgr*plot_scaling_y, label="Background")
    mplhep.histplot(np.histogram(data_projy, bins=n_bins), yerr=True, color='black', histtype='errorbar',
                    markersize=17, capsize=2.5,
                    markeredgewidth=1.5, zorder=1,
                    elinewidth=1.5, ax=axs[1]
                    )
    axs[1].set_title('Y projection')
    if fit_params_dict:
        assert 'y' in fit_params_dict
        assert 'mu' in fit_params_dict['y'] and 'sigma' in fit_params_dict['y'] and 'fr' in fit_params_dict['y']
        mu_patch = mpatches.Patch(color='none', label=f"mu = {fit_params_dict['y']['mu']:.2f}")
        sigma_patch = mpatches.Patch(color='none', label=f"sigma = {fit_params_dict['y']['sigma']:.2f}")
        fr_patch = mpatches.Patch(color='none', label=f"fr = {fit_params_dict['y']['fr']:.4f}")
        axs[1].legend(handles=[mu_patch, sigma_patch, fr_patch])
    if savefig:
        fig.savefig(f"fit_results/{image_name.split('.png')[0]}_fitted.png")
