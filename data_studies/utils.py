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

def extract_fit_features(minuit_obj, model, data, obs_bin_centers):
    fit_features = dict(minuit_obj.values)
    #
    total_count = fit_features['N']
    fr = fit_features['fr']
    sigma = fit_features['sigma']
    #
    fit_features['sig_count'] = total_count * fr
    fit_features['bkgr_count'] = total_count * (1-fr)
    fit_features['sig_density'] = total_count * fr / sigma
    fit_features['chi2'], fit_features['chi2_pvalue'] = chisquare(model(obs_bin_centers, **dict(minuit_obj.values)), data, ddof=4) # ddof hardcoded to number of fitted parameters
    fit_features['n_excess_bins'] = sum(data > total_count * (1-fr) / len(data))
    #
    fit_features['fr_error'] = dict(minuit_obj.errors)['fr']
    fit_features['mu_error'] = dict(minuit_obj.errors)['mu']
    fit_features['sigma_error'] = dict(minuit_obj.errors)['sigma']
    fit_features['N_error'] = dict(minuit_obj.errors)['N']
    #
    fit_features.update(dict(minuit_obj.fmin))
    return fit_features

def extract_fit_global_features(fit_features, ima):
    fit_global_features = {}
    # assert np.sum(ima['x']) == np.sum(ima['y'])
    # fit_global_features['total_count'] = np.sum(ima['x'])
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

def plot_projections(data_counts, model_prediction, data_bin_edges, model_prediction_grid, fit_params=None, savefig=True, output_folder='.', image_name=None):
    fig, axs = plt.subplots(1, 2, figsize=(20,7))
    data_counts_x = data_counts['x']
    data_counts_y = data_counts['y']
    #
    model_prediction_x = model_prediction['x']['model']
    model_prediction_x_sig = model_prediction['x']['sig']
    model_prediction_x_bkgr = model_prediction['x']['bkgr']
    #
    model_prediction_y = model_prediction['y']['model']
    model_prediction_y_sig = model_prediction['y']['sig']
    model_prediction_y_bkgr = model_prediction['y']['bkgr']
    ##
    axs[0].plot(model_prediction_grid, model_prediction_x, label="Model", linewidth=5)
    # axs[0].plot(model_prediction_grid, model_prediction_x_sig, label="Signal", linewidth=5)
    # axs[0].plot(model_prediction_grid, model_prediction_x_bkgr, label="Background", linewidth=5)
    mplhep.histplot(data_counts_x, data_bin_edges, yerr=True, color='black', histtype='errorbar',
                    markersize=17, capsize=2.5,
                    markeredgewidth=1.5, zorder=1,
                    elinewidth=1.5, ax=axs[0]
                    )
    axs[0].set_title('X projection')
    if fit_params:
        assert 'x' in fit_params
        assert 'mu' in fit_params['x'] and 'sigma' in fit_params['x'] and 'fr' in fit_params['x']
        mu_patch = mpatches.Patch(color='none', label=f"mu = {fit_params['x']['mu']:.2f}")
        sigma_patch = mpatches.Patch(color='none', label=f"sigma = {fit_params['x']['sigma']:.2f}")
        fr_patch = mpatches.Patch(color='none', label=f"fr = {fit_params['x']['fr']:.4f}")
        N_patch = mpatches.Patch(color='none', label=f"N = {fit_params['x']['N']:.0f}")
        axs[0].legend(handles=[mu_patch, sigma_patch, fr_patch, N_patch])
    ##
    axs[1].plot(model_prediction_grid, model_prediction_y, label="Model", linewidth=5)
    # axs[1].plot(model_prediction_grid, model_prediction_y_sig, label="Signal", linewidth=5)
    # axs[1].plot(model_prediction_grid, model_prediction_y_bkgr, label="Background", linewidth=5)
    mplhep.histplot(data_counts_y, data_bin_edges, yerr=True, color='black', histtype='errorbar',
                    markersize=17, capsize=2.5,
                    markeredgewidth=1.5, zorder=1,
                    elinewidth=1.5, ax=axs[1]
                    )
    axs[1].set_title('Y projection')
    if fit_params:
        assert 'y' in fit_params
        assert 'mu' in fit_params['y'] and 'sigma' in fit_params['y'] and 'fr' in fit_params['y']
        mu_patch = mpatches.Patch(color='none', label=f"mu = {fit_params['y']['mu']:.2f}")
        sigma_patch = mpatches.Patch(color='none', label=f"sigma = {fit_params['y']['sigma']:.2f}")
        fr_patch = mpatches.Patch(color='none', label=f"fr = {fit_params['y']['fr']:.4f}")
        N_patch = mpatches.Patch(color='none', label=f"N = {fit_params['y']['N']:.0f}")
        axs[1].legend(handles=[mu_patch, sigma_patch, fr_patch, N_patch])
    if savefig:
        fig.savefig(f"{output_folder}/{image_name.split('.png')[0]}_fitted.png")
    plt.close(fig)
