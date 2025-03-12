DEBUG = False

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import pandas as pd
import tensorflow as tf
import pickle

# tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC

# shap
import shap
import random
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Reshape

# AUCMEDI libraries
from aucmedi import input_interface, DataGenerator,Image_Augmentation
from aucmedi.neural_network.architectures import supported_standardize_mode
from aucmedi.data_processing.subfunctions import Padding
from aucmedi.ensembler import predict_augmenting

# Custom libraries
from retinal_crop import Retinal_Crop
from aucmedi_model import Neural_Network

import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

plt.rcParams["font.family"] = "Times New Roman"    
plt.rcParams["font.size"] = 14                  
plt.rcParams["xtick.direction"] = "in" 
plt.rcParams["ytick.direction"] = "in" 
plt.rcParams["xtick.minor.visible"] = True 
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0 
plt.rcParams["xtick.minor.width"] = 1.0 
plt.rcParams["ytick.minor.width"] = 1.0
plt.rcParams["xtick.major.size"] = 10
plt.rcParams["ytick.major.size"] = 10
plt.rcParams["xtick.minor.size"] = 5
plt.rcParams["ytick.minor.size"] = 5
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False
plt.rcParams["axes.linewidth"] = 1.0

plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1 
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.handlelength"] = 1 
plt.rcParams["legend.labelspacing"] = 5. 
plt.rcParams["legend.handletextpad"] = 3. 
plt.rcParams["legend.markerscale"] = 2 

# LaTeX for plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'


#-----------------------------------------------------#
#                   Configurations                    #
#-----------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#-----------------------------------------------------#
#              probability Rounding                   #
#-----------------------------------------------------#
def format_number_latex(a):
    if a >= 0.010:
        # Round to 3 decimal places to 2 decimal places
        rounded_prediction = round(a, 3)
        formatted_prediction = "{:.3f}".format(rounded_prediction)
        return formatted_prediction
    else:
        # Display as 3 digits in scientific notation and convert to LaTeX format
        sci_format = f"{a:.1e}"
        if 'e+' in sci_format:
            base, exponent = sci_format.split('e+')
            exponent = int(exponent)  # 指数部分を整数にして先頭のゼロを削除
            sci_format = f"{base} \\times 10^{{{exponent}}}"
        elif 'e-' in sci_format:
            base, exponent = sci_format.split('e-')
            exponent = int(exponent)  # 指数部分を整数にして先頭のゼロを削除
            sci_format = f"{base} \\times 10^{{-{exponent}}}"

        return f"${sci_format}$"
    
# Provide pathes to imaging data
if (DEBUG == False): path_riadd = "./circle_detection/images" 
else : path_riadd ="./debug" 

# Load shap_values
shap = pd.read_csv("./scripts/shap_scores.csv")


# Define parameters
k_fold = 5
processes = 8
batch_queue_size = 16
threads = 32

# Define label columns
cols = ["Disease_Risk", "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM",
        "LS", "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST",
        "AION", "PT", "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"]

# Create result directory
path_res = os.path.join("./preds")
if not os.path.exists(path_res) : os.mkdir(path_res)
# Obtain model directory
path_models = os.path.join("./models")

#-----------------------------------------------------#
#       General AUCMEDI Inference Pipeline Setup      #
#-----------------------------------------------------#
# Initialize input data reader
ds = input_interface(interface="directory", path_imagedir=path_riadd,
                     path_data=None, training=False)
(index_list, _, _, _, image_format) = ds
if '.DS_S' in index_list:
    index_list.remove('.DS_S')

# Define Subfunctions
sf_list = [Padding(mode="square"), Retinal_Crop()]

# Initialize Image Augmentation
aug = Image_Augmentation(flip=True, rotate=True, brightness=True, contrast=True,
                         saturation=False, hue=False, scale=False, crop=False,
                         grid_distortion=False, compression=False, gamma=False,
                         gaussian_noise=False, gaussian_blur=False,
                         downscaling=False, elastic_transform=False)

#-----------------------------------------------------#
#            AUCMEDI Classifier Inference             #
#-----------------------------------------------------#
# Define number of classes
nclasses = len(cols[1:])

# Set activation output to sigmoid for multi-label classification
activation_output = "sigmoid"

# Iterate over all classifier architectures
model_subdir = "classifier_EfficientNetB4"

# Identify architecture
arch = model_subdir.split("_")[1]
path_arch = os.path.join(path_models, model_subdir)

# Iterate over each fold of the CV
i = 4
model = Neural_Network(nclasses, channels=3, architecture=arch,
                        workers=processes,
                        batch_queue_size=batch_queue_size,
                        activation_output=activation_output,
                        loss="binary_crossentropy",
                        metrics=["binary_accuracy", AUC(100)],
                        pretrained_weights=True, multiprocessing=True,)

# Obtain standardization mode for current architecture
sf_standardize = supported_standardize_mode[arch]

# Define input shape
input_shape = model.input_shape[:-1]

index_list = [file for file in index_list if not file.startswith('c')]

# Initialize Data Generator for prediction
pred_gen = DataGenerator(index_list, path_riadd, labels=None,
                            batch_size=64, img_aug=None,
                            subfunctions=sf_list,
                            standardize_mode=sf_standardize,
                            shuffle=False, resize=input_shape,
                            grayscale=False, prepare_images=False,
                            sample_weights=None, seed=None,
                            image_format=image_format, workers=threads)

# Load best model
path_cv_model = os.path.join(path_arch, "cv_" + str(i) + ".model.best.hdf5")
if os.path.exists(path_cv_model) : model.load(path_cv_model)
else:
    print("Skipping model:", model_subdir, arch, str(i))

pred_gen_raw = DataGenerator(index_list, path_riadd, labels=None,
                            batch_size=64, img_aug=None,
                            subfunctions=sf_list,
                            standardize_mode=None,
                            shuffle=False, resize=input_shape,
                            grayscale=False, prepare_images=False,
                            sample_weights=None, seed=None,
                            image_format=image_format, workers=threads)
actmodel = model.actmodel()


df_probability = pd.read_csv("./auc for each classifier/EfficientNetB4/auc_classifier.EfficientNetB4.cv_4.inference.simple.csv")

df_probability= df_probability.drop("answer",axis = 1)

for j in range(len(index_list)):
    print(index_list[j])

    #image
    picture = pred_gen.preprocess_image(index = j, prepared_batch=False)
    picture = picture[np.newaxis,:,:,:]

    preds = actmodel.predict(picture)

    activation1 = preds[0][0]

    picture_raw = pred_gen_raw.preprocess_image(index = j, prepared_batch=False)[np.newaxis,:,:,:]

    image = picture_raw[0] / 255.

    shap_picture = shap[shap["ID"]  == index_list[j]]
    shap_value_picture = shap_picture.drop("ID",axis = 1).values[0]

    # Make 1792 heatmaps multiplied shap value
    importance_heatmaps = []
    for idx in range (len(preds[0][0][0,0,:])):
        heatmap = activation1[:,:,idx]
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy() * shap_value_picture[idx]
        importance_heatmaps.append(heatmap)

    ######################  Make heatmap ######################
    combined_heatmap = np.sum(importance_heatmaps, axis=0)
    combined_heatmap = combined_heatmap / np.max(combined_heatmap)
    combined_heatmap = resize(combined_heatmap, (380, 380), order=1, mode="reflect", anti_aliasing=False)


    ######################  Make heatmap (absolute value)#################
    combined_heatmap_abs = np.sum(importance_heatmaps, axis=0)
    combined_heatmap_abs = np.abs(combined_heatmap_abs)
    combined_heatmap_abs = combined_heatmap_abs / np.max(combined_heatmap_abs)
    combined_heatmap_abs = resize(combined_heatmap_abs, (380, 380), order=0, mode="reflect", anti_aliasing=False)


    ######################  prediction　#################
    probability = df_probability[df_probability["ID"] == index_list[j]]["probability"].tolist()[0]
    probability_latex = format_number_latex(probability)

    ######################  Make figure ######################
    # raw image
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 0.05, 1,0.05])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('[a] Input Image (probability:' + probability_latex + ')')

    # Heatmap (absolute value)
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(image)
    im = ax2.imshow(combined_heatmap_abs, cmap="jet", alpha=0.5)
    ax2.axis('off')
    ax2.set_title(f'[b] Feature Heatmap (Absolute Value, 0 to 1)')

    ax3 = fig.add_subplot(gs[0, 2])
    plt.colorbar(im, cax=ax3) 

    # Heatmap
    ax4 = fig.add_subplot(gs[0, 3])
    im = ax4.imshow(image)
    im = ax4.imshow(combined_heatmap, 
                    cmap="jet", 
                    norm= TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), 
                    alpha=0.6)
    ax4.axis('off')
    ax4.set_title(f'[c] Heatmap (Range -1 to 1)')

    ax5 = fig.add_subplot(gs[0, 4])
    plt.colorbar(im, cax=ax5)

    #Save figure
    plt.tight_layout()
    plt.savefig("./diagnosis_visualization/"+index_list[j]+"_diagnosis_visualization.pdf", bbox_inches='tight')
    plt.close(fig)