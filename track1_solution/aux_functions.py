import numpy as np
import pandas as pd
from scipy.stats import chisquare
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

def invert_regr(a, b, y):
    E_regr = (y - b)/a
    return E_regr

def fit_line(df):
    X_NR = df.query('event_energy>=3 and event_energy<=30 and event_class == "NR"').event_energy
    y_NR = df.query('event_energy>=3 and event_energy<=30 and event_class == "NR"').sig_count_y
    a_NR, b_NR = np.polyfit(X_NR, y_NR, deg=1)
    #
    xmin, xmax = 0, 40
    fig = plt.figure(figsize=(15,12))
    plt.scatter(X_NR, y_NR, label='true', alpha=0.3)
    plt.plot([xmin, xmax], a_NR*np.array([xmin, xmax])+b_NR, label='predicted')
    plt.xlim(xmin, xmax)
    # plt.ylim(-1000, 40000)
    plt.xlabel('true energy')
    plt.ylabel('sig_count_y')
    plt.legend(loc='upper center')
    plt.title(f'NR E<->sig_count_y calibration curve, a: {a_NR}, b: {b_NR}')
    plt.show()
    fig.savefig('calibration_curve_NR.png')
    return a_NR, b_NR

class Sigma_classifier:
    def __init__(self, sigma_y_th=5.3):
        self.sigma_y_th = sigma_y_th

    def predict(self, df_for_predict=[]):
        l = len(df_for_predict)
        df_data = df_for_predict.values.ravel()
        out = np.array([(df_data[i] - self.sigma_y_th) / 5.25 for i in range(l)])
        out[out > 1] = 1
        out[out < -1] = -1
        out += 1
        out /= 2
        out = 1 - out
        return list(out)
