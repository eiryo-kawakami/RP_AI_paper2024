import os
import json
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import pandas as pd
import tensorflow as tf
import pickle

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC

# AUCMEDI libraries
from aucmedi import input_interface, DataGenerator,Image_Augmentation
from aucmedi.neural_network.architectures import supported_standardize_mode
from aucmedi.data_processing.subfunctions import Padding
from aucmedi.ensembler import predict_augmenting

# Custom libraries
from retinal_crop import Retinal_Crop
from aucmedi_model import Neural_Network

#-----------------------------------------------------#
#                   Configurations                    #
#-----------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# Provide pathes to imaging data
path_riadd = "./circle_detection/images"

# Define some parameters
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

df_merged = pd.DataFrame()
for j in range (len(index_list)):
    f = open('./scripts/preds/preds_'+index_list[j]+'.txt', 'rb')
    value = pickle.load(f)
    df_pd = pd.DataFrame(data=value[1])
    df_pd.insert(0,"ID",index_list[j])
    df_merged = pd.concat([df_merged,df_pd])

columns = ["ID"]
for i in range(df_merged.shape[1]-1):
    column = "feature_" + str(i+1)
    columns.append(column)
df_merged.columns = columns

df_merged.to_csv("./scripts/extracted_features.csv",index=False)