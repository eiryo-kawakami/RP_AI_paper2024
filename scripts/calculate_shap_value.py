import os
import json
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import pandas as pd
import tensorflow as tf
import pickle

# tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC

# shap
import shap
import random
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Reshape

# keras
from keras.models import Model
from keras.layers import Dense, Lambda

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
path_riadd = "./circle_detection/images/"

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

pred_gen = DataGenerator(index_list, path_riadd, labels=None,
                            batch_size=64, img_aug=None,
                            subfunctions=sf_list,
                            standardize_mode=sf_standardize,
                            shuffle=False, resize=input_shape,
                            grayscale=False, prepare_images=False,
                            sample_weights=None, seed=None,
                            image_format=image_format, workers=threads)

# Load best model
path_cv_model = os.path.join("./"+path_arch, "cv_" + str(i) + ".model.best.hdf5")
model.load(path_cv_model)
model.summary()
model_real = model.acquire_model()


target_layer_name  = ["dense_2"]
shap_model = tf.keras.Model(
    [model_real.get_layer(target_layer_name[0]).input],
    [model_real.output]
)

# Get the second to last column
def extract_second_last_column(x):
    return x[:, -2:-1]  

# extract values before Dense block
existing_output = shap_model.output
new_layer = Lambda(extract_second_last_column)(existing_output)
new_shap_model = Model(inputs=shap_model.input, outputs=new_layer)


# data 
state_shap_input = pd.read_csv("./scripts/extracted_features.csv").drop("ID",axis = 1)

state_feature_names = state_shap_input.columns.values.tolist()
n_feature = len(state_feature_names) 
print('number of features =', n_feature)

row_shap_input = len(state_shap_input)
print('data length (time direction) =', row_shap_input)

state_shap_input = state_shap_input.values

state_SH_shap_input = state_shap_input[:,:]

print('shap SHAP value calc. input data shape =', state_SH_shap_input.shape)


# Calculate shap score
explainer = shap.DeepExplainer(new_shap_model, state_shap_input)

with_state_shap_values = explainer.shap_values(state_shap_input)

shap.summary_plot(
    with_state_shap_values[0],
    features=state_SH_shap_input,
    feature_names=state_feature_names,
    max_display=10,
    show=False,
    plot_type=None
)
plt.savefig("./scripts/shap_score.pdf")

shap.summary_plot(
    with_state_shap_values,
    features=state_SH_shap_input,
    feature_names=state_feature_names,
    max_display=10,
    show=False,
    plot_type=None
)
plt.savefig("./scripts/shap_score_bar.pdf")

df = pd.read_csv("./scripts/extracted_features.csv")

shap_scores_df = pd.DataFrame(with_state_shap_values[2], columns=state_feature_names)

shap_scores_df.insert(0, 'ID', df["ID"])

shap_scores_df.to_csv('./scripts/shap_scores.csv', index=False)
