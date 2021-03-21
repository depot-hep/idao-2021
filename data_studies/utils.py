import numpy as np
import pandas as pd
from scipy.stats import chisquare
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

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

def extract_global_features(image_path):
    global_features = {}
    if image_path.split('/')[-1].split('_')[5] == 'ER':
        global_features['event_class'] = 'ER'
        global_features['event_energy'] = image_path.split('/')[-1].split('_')[6]

    elif image_path.split('/')[-1].split('_')[6] == 'NR':
        global_features['event_class'] = 'NR'
        global_features['event_energy'] = image_path.split('/')[-1].split('_')[7]
    else:
        raise Exception("failed to infer event class")
    global_features['image_name'] = image_path.split('/')[-1].split(';1.png')[0]
    global_features['event_ID'] = image_path.split('/')[-1].split('_')[-1].split(';')[0][2:]
    global_features['event_angle'] = image_path.split('/')[-1].split('_')[0]
    return global_features

def extract_fit_features(model, data_proj_dict, ima_proj, obs_bin_centers, minimizer_results):
    fit_features = {p.name: p.numpy() for p in model.get_params()}
    #
    total_count = (data_proj_dict.n_events).numpy()
    fr = fit_features['fr']
    sigma = fit_features['sigma']
    background = model.get_models()[1]
    #
    fit_features['sig_count'] = total_count * fr
    fit_features['bkgr_count'] = total_count * (1-fr)
    fit_features['sig_density'] = total_count * fr / sigma
    fit_features['chi2'] = chisquare(sum(ima_proj)*model.pdf(obs_bin_centers), ima_proj, ddof=3)[0] # ddof hardcoded to number of fitted parameters
    fit_features['chi2_pvalue'] = chisquare(sum(ima_proj)*model.pdf(obs_bin_centers), ima_proj, ddof=3)[1] # ddof hardcoded to number of fitted parameters
    fit_features['n_excess_bins'] = sum(ima_proj > (background.pdf(obs_bin_centers)[0] * sum(ima_proj)).numpy())
    #
    fit_features.update(minimizer_results.info['original'])
    hesse_dict = dict(minimizer_results.hesse())
    fit_features['fr_error'] = list(hesse_dict.values())[0]['error']
    fit_features['mu_error'] = list(hesse_dict.values())[1]['error']
    fit_features['sigma_error'] = list(hesse_dict.values())[2]['error']
    #
    return fit_features

def extract_fit_global_features(fit_features, ima):
    fit_global_features = {}
    assert np.sum(ima['x']) == np.sum(ima['x'])
    fit_global_features['total_count'] = np.sum(ima['x'])
    fit_global_features['dmu'] = fit_features['x']['mu'] - fit_features['y']['mu']
    fit_global_features['dsigma'] = fit_features['x']['sigma'] - fit_features['y']['sigma']
    fit_global_features['dfr'] = fit_features['x']['fr'] - fit_features['y']['fr']
    return fit_global_features

def merge_proj_dict(fit_features):
    merged_fit_features = {}
    for proj in fit_features.keys():
        for feature in fit_features[proj].keys():
            merged_fit_features[f'{feature}_{proj}'] = fit_features[proj][feature]
    return merged_fit_features

def fill_dataframe(df, global_features, fit_features, fit2D_features, log_me=True, log_index=None, output_folder='.'):
    merged_fit_features = merge_proj_dict(fit_features)
    total_features = dict(global_features, **merged_fit_features, **fit2D_features)
    entry = pd.Series(total_features)
    df = df.append(entry, ignore_index=True)
    if log_me:
        if log_index != -1:
            df_name = f'super_puper_df_log_{log_index}.csv'
        else:
            df_name = f'super_puper_df.csv'
        df.to_csv(f'{output_folder}/{df_name}')
    return df

def plot_projections(obs_grid, n_bins, data_proj_dict, model_proj_dict, plot_scaling_dict, fit_params_dict, savefig=True, output_folder='.', image_name=None):
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
    axs[1].plot(obs_grid, model_projy*plot_scaling_y, label="Model")
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
        fig.savefig(f"{output_folder}/{image_name.split('.png')[0]}_fitted.png")
