import numpy as np
import pandas as pd
from scipy.stats import chisquare
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import mplhep
mplhep.set_style('CMS')

import cv2, os

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
    global_features['event_class'] = np.nan
    global_features['event_energy'] = np.nan
    global_features['image_name'] = np.nan
    global_features['event_ID'] = np.nan
    global_features['event_angle'] = np.nan

    if len(image_path.split('/')[-1].split('_')) < 2:
        global_features['image_name'] = image_path.split('/')[-1].split('.png')[0]
        return global_features
    else:
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

def extract_fit_features(minuit_obj, minuit_obj_bkgr_only, model_pred, model_bkgr_only_pred, data):
    fit_features = dict(minuit_obj.values)
    #
    total_count = fit_features['N']
    fr = fit_features['fr']
    sigma = fit_features['sigma']
    #
    fit_features['sig_count'] = total_count * fr
    fit_features['bkgr_count'] = total_count * (1-fr)
    fit_features['sig_density'] = total_count * fr / sigma
    fit_features['chi2'], fit_features['chi2_pvalue'] = chisquare(model_pred, data, ddof=4) # ddof hardcoded to number of fitted parameters
    fit_features['n_excess_bins'] = sum(data > total_count * (1-fr) / len(data))
    #
    fit_features['fr_error'] = dict(minuit_obj.errors)['fr']
    fit_features['mu_error'] = dict(minuit_obj.errors)['mu']
    fit_features['sigma_error'] = dict(minuit_obj.errors)['sigma']
    fit_features['N_error'] = dict(minuit_obj.errors)['N']
    fit_features.update(dict(minuit_obj.fmin))
    #
    fit_features['N_bkgr_only'] = dict(minuit_obj_bkgr_only.values)['N']
    fit_features['N_error_bkgr_only'] = dict(minuit_obj_bkgr_only.errors)['N']
    fit_features['chi2_bkgr_only'], fit_features['chi2_pvalue_bkgr_only'] = chisquare(model_bkgr_only_pred, data, ddof=1) # ddof hardcoded to number of fitted parameters
    for key, value in minuit_obj_bkgr_only.fmin.items():
        fit_features[f'{key}_bkgr_only'] = value
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

def plot_projections(data_counts, model_prediction, data_bin_edges, model_prediction_grid,
                     fit_params=None, close_image=False, save_fig=True, output_folder='.', image_name=None):
    fig, axs = plt.subplots(1, 2, figsize=(20,7))
    data_counts_x = data_counts['x']
    data_counts_y = data_counts['y']
    #
    model_prediction_x = model_prediction['x']['model']
    # model_prediction_x_sig = model_prediction['x']['sig']
    # model_prediction_x_bkgr = model_prediction['x']['bkgr']
    #
    model_prediction_y = model_prediction['y']['model']
    # model_prediction_y_sig = model_prediction['y']['sig']
    # model_prediction_y_bkgr = model_prediction['y']['bkgr']
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
    if save_fig:
        fig.savefig(f"{output_folder}/{image_name}_fitted.png")
    if close_image:
        plt.close(fig)

def plot_images(dataset_folder, class_to_plot=None, energy_to_plot=None,
                crop_images=True, crop_size = (100,100), standardize = False,
                max_num_images = 15, rand_seed=10):

    # parse image list
    img_names = os.listdir(dataset_folder)

    # train/test image folder
    if 'public_test' in dataset_folder or 'private_test' in dataset_folder:
        train_data = False
    else:
        train_data = True

    # select a subset of images of specific class and energy to be plotted
    if class_to_plot != None and energy_to_plot != None  and train_data:
        required_im_name = class_to_plot + '_' + str(energy_to_plot) + '_keV'
        im_fl = [required_im_name in img_names[idx] for idx in range(len(img_names))]
        img_names = [img_names[idx] for idx in range(len(img_names)) if im_fl[idx]]

    # permute images randomly
    num_images = len(img_names)
    np.random.seed(rand_seed)
    rand_permutation = np.random.permutation(num_images)
    img_names = [img_names[rand_permutation[i]] for i in range(num_images)]

    # create subplots
    nb_cols = 5
    nb_rows = (min(len(img_names), max_num_images) - 1) // nb_cols + 1
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(20, max(3*nb_rows,1)))
    if nb_rows == 1:
        axs = [axs]
    n = 0

    if crop_images:
        plotted_ims = np.zeros((max_num_images, crop_size[0], crop_size[1]))
    else:
        plotted_ims = np.zeros((max_num_images, 576, 576))


    # iterate over the selected images
    for i, img_name in enumerate(img_names):
        row_idx, col_idx = i // nb_cols, i - nb_cols*(i // nb_cols)

        # parse name (different for train and test images)
        if len(img_name.split('/')[-1].split('_')) < 2:
            label = img_name[:10] + '...'
        else:
            if img_name.split('/')[-1].split('_')[5] == 'ER':
                label = 'ER_' + img_name.split('/')[-1].split('_')[6] + '_keV (' + img_name[-13:] + ')'
            elif img_name.split('/')[-1].split('_')[6] == 'NR':
                label = 'NR_' + img_name.split('/')[-1].split('_')[7] + '_keV (' + img_name[-13:] + ')'
            else:
                label = 'Error'

        # read an image
        if dataset_folder[-1] != '/':
            dataset_folder += '/'
        img = cv2.imread(dataset_folder + img_name, cv2.IMREAD_COLOR)

        # standardize image if necessary
        img_plot = img[:,:,0]
        if standardize:
            img_plot = np.log(np.abs(((img_plot - np.mean(img_plot)))/np.std(img_plot)))

        # crop image if necessary
        if crop_images:
            cropped_w, cropped_h = crop_size[0] // 2, crop_size[1] // 2
            w, h = np.shape(img_plot)[0], np.shape(img_plot)[1]
            x_c, y_c = w // 2, h // 2
            img_plot = img_plot[x_c-cropped_w:x_c+cropped_w, y_c-cropped_h:y_c+cropped_h]

        plotted_ims[i, :, :] = img_plot

        # plot image
        axs[row_idx][col_idx].xaxis.set_ticklabels([])
        axs[row_idx][col_idx].yaxis.set_ticklabels([])
        axs[row_idx][col_idx].imshow(img_plot)
        axs[row_idx][col_idx].title.set_text(label)

        # break when max_num_images images have been plotted
        if i == max_num_images - 1:
            break

    plt.show()

    return plotted_ims
