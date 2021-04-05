# IDAO 2021: Baobab solution

This repository contains our team' solution to the qualification round of IDAO 2021, which resulted in the 5th place on Track 1. We carefully elaborate on the workflow of our approach in [this presentation](https://github.com/depot-hep/idao-2021/blob/main/Baobab-IDAO-2021-solution.pdf) - feel free to have a look there for getting a general understanding of the idea. In this README we just briefly describe the code's structure and further technical details on how to run it. 

The repo is structured as follows:
* `data_studies` contains all the necessary files to perform the fitting and plotting of images + archived studies we did on the way.
* `track1_solution` contains the code we used to produce predictions in Track 1 settings. 
* `track2_solution` contains the code we used to produce predictions in Track 2 settings. 
* `environment_idao.yml` describes the conda environment which was used
* `Baobab-IDAO-2021-solution.pdf` are the slides with the description of our approach

In order to run the fitting on images use `python extract_features.py fitting_cfg.yml`. In the `fitting_cfg.yml` file you'll be able to specify various input parameters to the script, e.g. the location of original images, output folders, init parameters of the fit, etc. In `fit_model.py` the fitting model and custom chi^2 minimization procedure are described, and in `utils.py` we collect various helper functions for e.g. filling the output dataframe. The fitting procedure itself outputs two things:
1) `.csv` files storing `pandas.DataFrame` with all the features obtained from the fit -> further used to make predictions
2) plots illustrating results of the fit (data histograms in two projections with overlayed fitting curve with its parameters) -> [here](https://disk.yandex.ru/d/iQ2bUKYVfjaP1w) you can find the corresponding plots which we obtained in the context of our solution. 

Once the fit is done and output `.csv` files (see `data_studies/csv_data` for those we used) are obtained, one can proceed to the prediction step. Submission file with corresponding predictions is produced by running through all the cells of `track1_solution/pipeline.ipynb` notebook. Please have a look there and into the presentation to get more details about the actual inference step. Also note that `region_params.yml` file contains the configuration parameters required by the prediction procedure. Last but not the least, it's worth mentioning `aux_functions.py` where we keep some helper functions, most notably the one which produces the calibration curve.

As for `track2_solution`, the structure inherits from the sample submission and `generate_submission.py` contains all the necessary code for producing submission predictions.

## Baobab team
* Oleg Filatov [DESY] (<oleg.filatov@phystech.edu>), team leader
* Andrey Znobishchev [Skoltech] (<andrei.znobishchev@skoltech.ru>) 
* Andrei Filatov [MIPT, EPFL] (<filatov.av@phystech.edu>)
