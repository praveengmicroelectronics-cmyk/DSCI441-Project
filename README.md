# DSCI441-Project
# Milestone1

# new file:   Feature_Extraction.py, new file:   Feature_parameters.py, new file:   Grid_Generation.py, new file:   LSWMD.pkl, new file:   Milestone1.py, new file:   Wafer_Defect_Classification.mp4, new file:   Wafer_Defect_Classification.pptx, new file:   plot_maps.py 
# Please download LSWMD.pkl data file from https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/data

# Abstract: Milestone1
Semiconductor manufacturing requires high production yield and reliability, and wafer defects significantly impact both product quality and manufacturing efficiency. Defects appearing on wafer maps often follow identifiable spatial patterns such as localized clusters, edge defects, ring patterns, scratches, and random distributions. In many semiconductor fabrication facilities, defect classification from wafer maps is still performed manually, which can lead to inconsistencies and subjective interpretation. The goal of this project is to develop an automated defect classification system using statistical machine learning methods.

The project uses a wafer map dataset containing wafer images, defect labels, and die size information. The dataset is preprocessed by removing missing labels, excluding the “none” defect category, and filtering wafers based on die size to ensure consistent feature extraction. Instead of using deep learning approaches, this work focuses on extracting interpretable statistical features that capture spatial characteristics of defect patterns. These features include defect density, centroid location, radial distribution statistics, edge defect ratios, quadrant distributions, spread, ring detection metrics, and cluster density. Grid coordinates are normalized relative to the wafer center so that spatial features can be computed consistently across wafers.

Using these extracted features, a baseline classification model is implemented with a train–test split of 80/20. Model performance is evaluated using accuracy, balanced accuracy, class-wise recall, and confusion matrices to assess the classifier’s ability to distinguish different defect categories. Preliminary results provide an initial benchmark for automated defect classification. In the next milestone, additional statistical learning models such as Logistic Regression, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), and Support Vector Machines (SVM) will be implemented and compared. Model performance will be analyzed using training and testing errors as a function of model complexity to study the bias–variance trade-off and identify the most suitable classifier for wafer defect detection.

Ultimately, this work aims to provide an interpretable machine learning framework for wafer defect classification that can support semiconductor fabrication facilities in identifying potential process or equipment issues more efficiently.


###Step2########
Feature Selection
Loads the raw `.pkl` dataset from the path. Flattens the `failureType` column, drops rows with missing labels or die sizes, filters to `dieSize < 2500`, and returns the cleaned DataFrame.

Calls _load_df, then applies extract_wafer_features` (from `Feature_Extraction.py`) to every wafer map using parameters from `get_feature_params()`. Returns a DataFrame of **34 features** per wafer. 

Returns the `failureType` label column from `_load_df`, reset to a default integer index so it aligns with the feature matrix.

1. Calls `_extract_features` and `_get_labels` to get `X` and `y`.
2. Cleans `X` (fills NaN / ±inf with 0).
3. Performs a stratified 80/20 train/test split using `random_state`.
4. Fits a `RandomForestClassifier` (3000 trees, `log2` max features, balanced class weights) on the **training split only** — no label leakage from the test set.
5. Ranks all 34 features by mean decrease in impurity (MDI).
6. Returns a DataFrame with columns **Rank**, **Feature**, **Importance** for the top n_top features.

###################

