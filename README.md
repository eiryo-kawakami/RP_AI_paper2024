# Leveraging Large-Scale Deep Learning Models for Diagnosis and Visual Outcome Prediction in Retinitis Pigmentosa Patients

<!-- ## Prediction Model Overview
コメントか図を入れる。 -->

##  Strage for your data 
You have to save your dataset in `"storage/data"`.
Also, you have to prepare for `"storage/data_labels`" as a label data.

## Circle Detection
In our dataset, the top and bottom of image were cropped, so we adjusted the position of the fundus in the images using circle detection to match the positional relationship with the fundus in previous studies.
1. You can save the image size to a CSV file and find out how many different image sizes there are by running `"circle_detection/scripts/Image_size.ipynb"`.
2. You can determine the size and position of the fundus circle form each image size by running  `"Circle_detection/scripts/determine_circle_detection.ipynb"`.
For each unique size of fundus image, you must determine the circle that best fits the fundus. Therefore, you need to run this code separately for every size of fundus image.
3. Using steps 1 and 2, you can draw a circle that corresponds to the fundus by running `"circle_detection/scripts/circle_detection.ipynb"`. 

## Prediction and analysis
1. You can make diagnostic predictions using `"scripts/inference.py"` and `"scripts/ensemble.py"`. You can do this by simply `"running prediction.sh`"
   
2. You can also assess each classifier's performance using `"auc_for_each_classifier/comparison_of_each_classifier/ROC-AUC.ipynb"` and `"auc_for_each_classifier/comparison_of_each_classifier/Precision_Recall.ipynb"`.

## Extract Feature Value and visualization
You can do this section by running `"Extract Feature Value and visualization`"
1. You need to save predictions for each figure using `"scripts/preds_nobatch.py"`.
2. You can extract feature values after `"top_activation"`, `"avg_pool"`, and `"dropout_2"` using `"scripts/extract_values.py"`.
3. You can calculate SHAP values for diagnostic predictions using `"scripts/calculate_shap_value.py"` and visualize them using `"scripts/diagnosis_visualization.py"`.


## Prognostic Visualization
You can do this sevtion by runnning `"Prognostic visualization.sh`"
1. To visualize the prognosis prediction, you need to calculate the mean of survival SHAP values by running `"scripts/surv_shap_calculate_mean.py"`.
2. You can visualize prognostic predictions using `"scripts/prognosis_visualization.py"`.

