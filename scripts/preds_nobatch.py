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
# select fold of the CV
i = 4

# Define label columns
cols = ["Disease_Risk", "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM",
        "LS", "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST",
        "AION", "PT", "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"]

# Create result directory
path_res = os.path.join("preds")
if not os.path.exists(path_res) : os.mkdir(path_res)
# Obtain model directory
path_models = os.path.join("models")

#-----------------------------------------------------#
#       General AUCMEDI Inference Pipeline Setup      #
#-----------------------------------------------------#
# Initialize input data reader
ds = input_interface(interface="directory", path_imagedir=path_riadd,
                     path_data=None, training=False)
(index_list, _, _, _, image_format) = ds
if '.DS_S' in index_list: index_list.remove('.DS_S')

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
model.load(path_cv_model)

model_real = model.acquire_model()

target_layer_name  = ["top_activation","avg_pool","dropout_2"]
grad_model = tf.keras.Model(
    [model_real.inputs],
    [model_real.get_layer(target_layer_name[0]).output,
     model_real.get_layer(target_layer_name[1]).output,
     model_real.get_layer(target_layer_name[2]).output,
     model_real.output]
)
print(len(index_list)) 

# Save prediction
for j in range(len(index_list)):
        print([j,index_list[j]])
        picture = pred_gen.preprocess_image(index = j, prepared_batch=False)
        picture = picture[np.newaxis,:,:,:]
        y = grad_model(picture)
        f = open('scripts/preds/preds_'+index_list[j]+'.txt', 'wb')
        pickle.dump(y, f)