import numpy as np
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

from utils import counts_to_datapoints, build_model, plot_projections

############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("cfg", type=str, help="path to yaml configuration file")
args = parser.parse_args()
with open(args.cfg) as f:
    cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

# definitions
path_to_images = cfg_dict['path_to_images']
images_range = cfg_dict['images_range']
assert images_range == -1 or (len(images_range)==2 and images_range[0]<=images_range[1])
if images_range == -1:
    image_paths = glob(path_to_images)
else:
    image_paths = glob(path_to_images)[images_range[0]:images_range[1]]
obs_left, obs_right = cfg_dict['obs']['range']
obs_grid = np.linspace(obs_left, obs_right, 1000)
n_bins = cfg_dict['obs']['n_bins']
mu_init = cfg_dict['init_values']['mu']
sigma_init = cfg_dict['init_values']['sigma']
fr_init = cfg_dict['init_values']['fr']

# build the model
obs = zfit.Space("obs", limits=(obs_left, obs_right))
model = build_model(obs, obs_left, obs_right, cfg_dict['init_values'])

# loop over images
for image_path in image_paths:
    im = Image.open(image_path)
    ima = np.array(im)
    ima_x = np.sum(ima[obs_left:obs_right, obs_left:obs_right], axis=0) # x projection
    ima_y = np.sum(ima[obs_left:obs_right, obs_left:obs_right], axis=1) # y projection
    x_samples, y_samples = counts_to_datapoints(ima_x, ima_y, obs_left, obs_right) # make data set out of bin counts

    # define observable to be fitted and project the data there
    data_proj_dict = {'x': zfit.data.Data.from_numpy(obs=obs, array=x_samples), 'y': zfit.data.Data.from_numpy(obs=obs, array=y_samples)}
    data_proj_np_dict = {'x': data_proj_dict['x'][:, 0].numpy(), 'y': data_proj_dict['y'][:, 0].numpy()}

    # scaling constant for plotting
    plot_scaling_dict = {'x': len(x_samples) / n_bins * obs.area(), 'y': len(y_samples) / n_bins * obs.area()}

    # fit projections
    fit_params_dict, model_proj_dict = {}, {}
    for proj in ['x', 'y']:
        model.get_params()[0].set_value(fr_init) # fraction
        model.get_params()[1].set_value(mu_init) # mu
        model.get_params()[2].set_value(sigma_init) # sigma
        minimizer = zfit.minimize.Minuit(verbosity=7)
        # minimizer = zfit.minimize.Adam() # but there's more
        nll = zfit.loss.UnbinnedNLL(model=model, data=data_proj_dict[proj])
        result = minimizer.minimize(nll) # , params=[mu, frac]
        result = minimizer.minimize(nll) # , params=[sigma, frac]
        result = minimizer.minimize(nll)
        # print(result.params)
        # print(result_x.valid)  # check if the result is still valid
        fit_params_dict[proj] = {p.name: p.numpy() for p in model.get_params()}
        model_proj_dict[proj] = {m.name: m.pdf(obs_grid).numpy() for m in model.get_models()}
        model_proj_dict[proj]['model'] = model.pdf(obs_grid).numpy()
    image_name = image_path.split('/')[-1].split(';1.png')[0]
    plot_projections(obs_grid, n_bins, data_proj_np_dict, model_proj_dict, plot_scaling_dict, fit_params_dict, savefig=True, image_name=image_name)
    gc.collect()
