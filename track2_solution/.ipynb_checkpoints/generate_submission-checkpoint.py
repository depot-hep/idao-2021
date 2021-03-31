import configparser
import pathlib as path
import cv2
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def img_loader(path: str):
    with Image.open(path) as img:
        img = np.array(img)
    return img

def crop_center(img, crop_size=(30,30)):
    z_c = (288, 288)
    half_w, half_h = crop_size[0] // 2, crop_size[1] // 2
    img_cropped = img[z_c[0]-half_w:z_c[0]+half_w, z_c[1]-half_h:z_c[1]+half_h]
    return img_cropped

def crop_corners(img, crop_size=(30,30)):
    z_c = (288, 288)
    half_w, half_h = crop_size[0] // 2, crop_size[1] // 2
    img_cropped1 = img[:2*half_w, :2*half_h]
    img_cropped2 = img[:2*half_w, 2*z_c[1]-2*half_h:2*z_c[1]]
    img_cropped3 = img[2*z_c[0]-2*half_w:2*z_c[0], :2*half_h]
    img_cropped4 = img[2*z_c[0]-2*half_w:2*z_c[0], 2*z_c[1]-2*half_h:2*z_c[1]]
    
    img_cropped = np.zeros((4, crop_size[0], crop_size[1]))
    img_cropped[0, :] = img_cropped1
    img_cropped[1, :] = img_cropped2
    img_cropped[2, :] = img_cropped3
    img_cropped[3, :] = img_cropped4
    return img_cropped

def proj_x(img):
    return np.sum(img, axis=0)

def proj_y(img):
    return np.sum(img, axis=1)

def mu(f):
    out = np.argwhere(f == np.max(f)).ravel()
    if len(out) > 1:
        out = np.min(np.abs(out - 50)) + 50
    else:
        out = out[0]
        
    if out > 5 and out < 95:
        f_peak = np.mean(f[out-5:out+5])
    else:
        f_peak = f[out]
                         
    return out, f_peak

def generate_submission(cl_pred, reg_pred, im_names, submission_name):
    
    # create dataframe
    submission = pd.DataFrame(columns=['id', 'classification_predictions', 'regression_predictions'])
    submission['id'] = im_names
    submission['classification_predictions'] = cl_pred
    submission['regression_predictions'] = reg_pred

    # sort values by id
    idx = submission.index
    submission.sort_values('id', inplace=True)
    submission.index = idx
    
    # save to csv
    submission.to_csv(submission_name + '.csv', index=False, header=True)


def main(cfg):
    
    crop_size = (100,100)
    x_c = crop_size[0] // 2
    f_th = 150
    z_c = (288, 288)
    a_NR = 33.1
    b_NR = 102.1
    th_dmu = 2
    th_sigma_MID = 13.5
    th_sigma_LOW = 1.99

    PATH = str(cfg["DATA"]["DatasetPath"])
    data_folder_NR = PATH + '/train/NR/'
    data_folder_ER = PATH +'/train/ER/'
    data_folder_public = PATH +'/public_test/'
    data_folder_private = PATH +'/private_test/'
    
    num_im_public = len(list(listdir_nohidden(data_folder_public)))
    num_im_private = len(list(listdir_nohidden(data_folder_private)))

    df_public = pd.DataFrame(index=np.arange(0, num_im_public), columns=['sig_count_y', 'sigma_y', 'abs_dmu_x', 'abs_dmu_y', 'image_name'])
    df_private = pd.DataFrame(index=np.arange(0, num_im_private),columns=['sig_count_y', 'sigma_y', 'abs_dmu_x', 'abs_dmu_y', 'image_name'])
    
    df_cur = pd.DataFrame(df_public)
    data_folder = data_folder_public

    files = list(listdir_nohidden(data_folder))
    for i in range(len(files)):
        im_name = files[i]
        img = img_loader(data_folder + im_name)

        center = crop_center(img, crop_size=crop_size)
        corners = crop_corners(img, crop_size=crop_size)
        bckg = (np.sum(np.mean(corners, axis=0), axis=0))
        f_x, f_y = proj_x(center)-bckg, proj_y(center)-bckg
        mu_x, f_x_max = mu(f_x)
        mu_y, f_y_max = mu(f_y)

        df_cur.iloc[i]['sig_count_y'] = f_y_max
        df_cur.iloc[i]['sigma_y'] = sum(f_y > f_th)
        df_cur.iloc[i]['abs_dmu_x'], df_cur.iloc[i]['abs_dmu_y'] = np.abs(mu_x - x_c), np.abs(mu_y - x_c)
        df_cur.iloc[i]['image_name'] = im_name[:-4]

    df_public = pd.DataFrame(df_cur)
    
    
    df_cur = pd.DataFrame(df_private)
    data_folder = data_folder_private

    files = list(listdir_nohidden(data_folder))
    for i in range(len(files)):
        im_name = files[i]
        img = img_loader(data_folder + im_name)

        center = crop_center(img, crop_size=crop_size)
        corners = crop_corners(img, crop_size=crop_size)
        bckg = (np.sum(np.mean(corners, axis=0), axis=0))
        f_x, f_y = proj_x(center)-bckg, proj_y(center)-bckg
        mu_x, f_x_max = mu(f_x)
        mu_y, f_y_max = mu(f_y)

        df_cur.iloc[i]['sig_count_y'] = f_y_max
        df_cur.iloc[i]['sigma_y'] = sum(f_y > f_th)
        df_cur.iloc[i]['abs_dmu_x'], df_cur.iloc[i]['abs_dmu_y'] = np.abs(mu_x - x_c), np.abs(mu_y - x_c)
        df_cur.iloc[i]['image_name'] = im_name[:-4]
    
    df_private = pd.DataFrame(df_cur)
    
    idx_HE_ER_publ = [i for i in range(len(df_public)) if (df_public.iloc[i]['sig_count_y'] - b_NR)/a_NR >=10 and df_public.iloc[i]['abs_dmu_y'] > th_dmu]
    idx_HE_NR_publ = [i for i in range(len(df_public)) if (df_public.iloc[i]['sig_count_y'] - b_NR)/a_NR >=10 and df_public.iloc[i]['abs_dmu_y'] <= th_dmu]

    idx_MID_ER_publ = [i for i in range(len(df_public)) if (df_public.iloc[i]['sig_count_y'] - b_NR)/a_NR < 10 and (df_public.iloc[i]['sig_count_y'] - b_NR)/a_NR > 3 and df_public.iloc[i]['sigma_y'] > th_sigma_MID]
    idx_MID_NR_publ = [i for i in range(len(df_public)) if (df_public.iloc[i]['sig_count_y'] - b_NR)/a_NR < 10 and (df_public.iloc[i]['sig_count_y'] - b_NR)/a_NR > 3 and df_public.iloc[i]['sigma_y'] <= th_sigma_MID]

    idx_LOW_ER_publ = [i for i in range(len(df_public)) if (df_public.iloc[i]['sig_count_y'] - b_NR)/a_NR <= 3 and df_public.iloc[i]['sigma_y'] > th_sigma_LOW]
    idx_LOW_NR_publ = [i for i in range(len(df_public)) if (df_public.iloc[i]['sig_count_y'] - b_NR)/a_NR <= 3 and df_public.iloc[i]['sigma_y'] <= th_sigma_LOW]
    
    im_name_publ = df_public['image_name'].values
    
    reg_pred_publ = np.array([0 for i in range(len(df_public))])
    reg_pred_publ[idx_LOW_NR_publ] = 1
    reg_pred_publ[idx_LOW_ER_publ] = 3
    reg_pred_publ[idx_MID_NR_publ] = 6
    reg_pred_publ[idx_MID_ER_publ] = 10
    reg_pred_publ[idx_HE_NR_publ] = 20
    reg_pred_publ[idx_HE_ER_publ] = 30

    cl_pred_publ = np.array([-1 for i in range(len(df_public))])
    cl_pred_publ[idx_LOW_NR_publ] = 0
    cl_pred_publ[idx_LOW_ER_publ] = 1
    cl_pred_publ[idx_MID_NR_publ] = 0
    cl_pred_publ[idx_MID_ER_publ] = 1
    cl_pred_publ[idx_HE_NR_publ] = 0
    cl_pred_publ[idx_HE_ER_publ] = 1
    
    im_name_priv = df_private['image_name'].values
    
    idx_HE_ER_priv = [i for i in range(len(df_private)) if (df_private.iloc[i]['sig_count_y'] - b_NR)/a_NR >=14 and df_private.iloc[i]['abs_dmu_y'] > th_dmu]
    idx_HE_NR_priv = [i for i in range(len(df_private)) if (df_private.iloc[i]['sig_count_y'] - b_NR)/a_NR >=14 and df_private.iloc[i]['abs_dmu_y'] <= th_dmu]

    idx_MID_ER_priv = [i for i in range(len(df_private)) if (df_private.iloc[i]['sig_count_y'] - b_NR)/a_NR < 14 and (df_private.iloc[i]['sig_count_y'] - b_NR)/a_NR > 6 and df_private.iloc[i]['sigma_y'] > th_sigma_MID]
    idx_MID_NR_priv = [i for i in range(len(df_private)) if (df_private.iloc[i]['sig_count_y'] - b_NR)/a_NR < 14 and (df_private.iloc[i]['sig_count_y'] - b_NR)/a_NR > 6 and df_private.iloc[i]['sigma_y'] <= th_sigma_MID]

    idx_LOW_ER_priv = [i for i in range(len(df_private)) if (df_private.iloc[i]['sig_count_y'] - b_NR)/a_NR <= 6 and df_private.iloc[i]['sigma_y'] > th_sigma_LOW]
    idx_LOW_NR_priv = [i for i in range(len(df_private)) if (df_private.iloc[i]['sig_count_y'] - b_NR)/a_NR <= 6 and df_private.iloc[i]['sigma_y'] <= th_sigma_LOW]
    
    reg_pred_priv = np.array([0 for i in range(len(df_private))])
    reg_pred_priv[idx_LOW_ER_priv] = 1
    reg_pred_priv[idx_LOW_NR_priv] = 3
    reg_pred_priv[idx_MID_ER_priv] = 6
    reg_pred_priv[idx_MID_NR_priv] = 10
    reg_pred_priv[idx_HE_ER_priv] = 20
    reg_pred_priv[idx_HE_NR_priv] = 30

    cl_pred_priv = np.array([-1 for i in range(len(df_private))])
    cl_pred_priv[idx_LOW_ER_priv] = 1
    cl_pred_priv[idx_LOW_NR_priv] = 0
    cl_pred_priv[idx_MID_ER_priv] = 1
    cl_pred_priv[idx_MID_NR_priv] = 0
    cl_pred_priv[idx_HE_ER_priv] = 1
    cl_pred_priv[idx_HE_NR_priv] = 0
    
    cl_pred = np.append(cl_pred_publ, cl_pred_priv)
    reg_pred = np.append(reg_pred_publ, reg_pred_priv)
    im_names = np.append(im_name_publ, im_name_priv)

    assert(len(im_names) == len(reg_pred))
    assert(len(im_names) == len(cl_pred))
    assert(len(im_names) == len(df_private) + len(df_public))

    generate_submission(cl_pred, reg_pred, im_names, 'submission')
    
if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
