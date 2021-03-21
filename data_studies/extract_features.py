import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import argparse
from glob import glob
import gc
import yaml

import zfit
import mplhep
mplhep.set_style('CMS')

from utils import *
from fit_model import build_model

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
obs_left, obs_right = cfg_dict['obs']['range']
obs_edges = np.array(range(obs_left, obs_right + 1))
obs_bin_centers = (obs_edges[:-1] + obs_edges[1:])/2
obs_grid = np.linspace(obs_left, obs_right, 1000)
n_bins = cfg_dict['obs']['n_bins']

mu_init = cfg_dict['init_values']['mu']
sigma_init = cfg_dict['init_values']['sigma']
fr_init = cfg_dict['init_values']['fr']

# build the model
obs = zfit.Space("obs", limits=(obs_left, obs_right))
model = build_model(obs, obs_left, obs_right, cfg_dict['init_values'])

# output dataframe
df = pd.DataFrame()

# loop over images
for image_index, image_path in enumerate(image_paths):
    log_me = not (image_index%log_step) or (image_index == len(image_paths)-1)
    log_index = image_index if image_index != len(image_paths)-1 else -1
    #
    im = Image.open(image_path)
    ima = {proj: np.sum(np.array(im)[obs_left:obs_right, obs_left:obs_right], axis=ax) for proj, ax in zip(['x', 'y'], [0, 1])} # image projections
    x_samples, y_samples = counts_to_datapoints(ima['x'], ima['y'], obs_left, obs_right) # make data set out of bin counts

    # define observable to be fitted and project the data there
    data_proj_dict = {'x': zfit.data.Data.from_numpy(obs=obs, array=x_samples), 'y': zfit.data.Data.from_numpy(obs=obs, array=y_samples)}
    data_proj_np_dict = {'x': data_proj_dict['x'][:, 0].numpy(), 'y': data_proj_dict['y'][:, 0].numpy()}

    # scaling constant for plotting
    plot_scaling_dict = {'x': len(x_samples) / n_bins * obs.area(), 'y': len(y_samples) / n_bins * obs.area()}

    # fit projections
    minimizer_results, model_proj_dict, fit_features, global_features = {}, {}, {}, {}
    for proj in ['x', 'y']:
        model.get_params()[0].set_value(fr_init) # fraction
        model.get_params()[1].set_value(mu_init) # mu
        model.get_params()[2].set_value(sigma_init) # sigma
        minimizer = zfit.minimize.Minuit(verbosity=fit_verbosity)
        # minimizer = zfit.minimize.Adam() # but there's more
        nll = zfit.loss.UnbinnedNLL(model=model, data=data_proj_dict[proj])
        minimizer_results = minimizer.minimize(nll) # , params=[mu, frac]
        minimizer_results = minimizer.minimize(nll) # , params=[sigma, frac]
        minimizer_results = minimizer.minimize(nll)
        #
        model_proj_dict[proj] = {m.name: m.pdf(obs_grid).numpy() for m in model.get_models()}
        model_proj_dict[proj]['model'] = model.pdf(obs_grid).numpy()
        fit_features[proj] = extract_fit_features(model, data_proj_dict[proj], ima[proj], obs_bin_centers, minimizer_results)
    fit2D_features = extract_fit_global_features(fit_features, ima)
    global_features = extract_global_features(image_path)
    df = fill_dataframe(df, global_features, fit_features, fit2D_features, log_me=log_me, log_index=log_index, output_folder=output_folder_data)
    plot_projections(obs_grid, n_bins, data_proj_np_dict, model_proj_dict, plot_scaling_dict, fit_features,
                     savefig=True, output_folder=output_folder_images, image_name=global_features['image_name'])
    gc.collect()
