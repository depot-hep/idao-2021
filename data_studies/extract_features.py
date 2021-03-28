import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import argparse
from glob import glob
import gc
import yaml

from utils import *
from fit_model import norm_pdf, uniform_pdf, model, model_bkgr_only, fit_model
from iminuit import Minuit
import mplhep
mplhep.set_style('CMS')

import time

############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("cfg", type=str, help="path to yaml configuration file")
args = parser.parse_args()
with open(args.cfg) as f:
    cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

# definitions
path_to_images = cfg_dict['path_to_images']
images_range = cfg_dict['images_range']
log_step = cfg_dict['log_step']
fit_verbosity = cfg_dict['fit_verbosity']
output_folder_data = cfg_dict['output_folder_data']
output_folder_images = cfg_dict['output_folder_images']
assert images_range == -1 or (len(images_range)==2 and images_range[0]<=images_range[1])
if images_range == -1:
    image_paths = glob(path_to_images)
else:
    image_paths = glob(path_to_images)[images_range[0]:images_range[1]]
obs_left, obs_right = cfg_dict['obs_range']
obs_edges = np.array(range(obs_left, obs_right + 1))
obs_bin_centers = (obs_edges[:-1] + obs_edges[1:])/2
obs_grid = np.linspace(obs_left, obs_right, 1000)

mu_init = cfg_dict['init_values']['mu']
sigma_init = cfg_dict['init_values']['sigma']
fr_init = cfg_dict['init_values']['fr']

# output dataframe
df = pd.DataFrame()

t_0 = time.time()

# loop over images
for image_index, image_path in enumerate(image_paths):
    if image_index % 1000 == 0:
    	print('Image: ', image_index)

    log_me = not (image_index%log_step) or (image_index == len(image_paths)-1)
    log_index = image_index if image_index != len(image_paths)-1 else -1
    #
    im = Image.open(image_path)
    ima = {proj: np.sum(np.array(im)[obs_left:obs_right, obs_left:obs_right], axis=ax) for proj, ax in zip(['x', 'y'], [0, 1])} # image projections

    # fit projections
    minimizer_results, fit_features, global_features = {}, {}, {}
    model_prediction_grid, model_bkgr_only_prediction_grid = {'x': {}, 'y': {}}, {'x': {}, 'y': {}}
    fit_OK = True
    for proj in ['x', 'y']:
        m, migrad_OK, hesse_OK = fit_model(model, obs_bin_centers, ima[proj], image_path, fit_verbosity,
                              mu=mu_init,
                              sigma=sigma_init,
                              fr=fr_init,
                              N=sum(ima[proj]),)
        m_bkgr_only, migrad_bkgr_only_OK, hesse_bkgr_only_OK = fit_model(model_bkgr_only, obs_bin_centers, ima[proj], image_path, fit_verbosity,
                                                  N=sum(ima[proj]))
        fit_OK *= (migrad_OK & hesse_OK & migrad_bkgr_only_OK & hesse_bkgr_only_OK)

        if fit_OK:
            # these are for chi2 calculation
            model_prediction = model(obs_bin_centers, **dict(m.values))
            model_bkgr_only_prediction = model_bkgr_only(obs_bin_centers, **dict(m_bkgr_only.values))
            fit_features[proj] = extract_fit_features(m, m_bkgr_only, model_prediction, model_bkgr_only_prediction, ima[proj])
            # these are for plotting
            model_prediction_grid[proj]['model'] = model(obs_grid, **dict(m.values))
            model_bkgr_only_prediction_grid[proj]['model'] = model_bkgr_only(obs_grid, **dict(m_bkgr_only.values))
            # model_prediction_grid[proj]['sig'] = fit_param_values['N']*fit_param_values['fr']*norm_pdf(obs_grid, fit_param_values['mu'], fit_param_values['sigma'])
            # model_prediction_grid[proj]['bkgr'] = fit_param_values['N']*(1-fit_param_values['fr'])*uniform_pdf(obs_grid)

    if fit_OK:
       fit_global_features = extract_fit_global_features(fit_features, ima)
       global_features = extract_global_features(image_path)
       df = fill_dataframe(df, global_features, fit_features, fit_global_features, log_me=log_me, log_index=log_index, output_folder=output_folder_data)
       plot_projections(ima, model_prediction_grid, obs_edges, obs_grid, fit_params=fit_features,
                     close_image=True, save_fig=True, output_folder=output_folder_images, image_name=global_features['image_name'].split('.png')[0])
       plot_projections(ima, model_bkgr_only_prediction_grid, obs_edges, obs_grid, #fit_params=fit_features,
                     close_image=True, save_fig=True, output_folder=output_folder_images, image_name=global_features['image_name'].split('.png')[0]+'_bkgr_only')
       gc.collect()

print('time: ', time.time() - t_0)
