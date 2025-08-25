# Databricks notebook source
# MAGIC %md
# MAGIC # Research Question: What are the optimal process specifications to satisfy optimal physical powder properties (PPP) for the SKU YCF325?

# COMMAND ----------

# MAGIC %md
# MAGIC ## This notebook is used for decision tree modelling for PPP Scorched Particles.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read multiple Parquet files from ADLS into Pandas DataFrames for modelling tasks.

# COMMAND ----------

# Define the container and storage account details
produced_container_name = "produced"
storage_account_name = dbutils.secrets.get(scope="DAN-AP-T-KVT800-R-OCE-DB", key="databricks-dev-storage-account-name")
source_store_directory = dbutils.secrets.get(scope="DAN-AP-T-KVT800-R-OCE-DB", key="databricks-dev-source-store-directory")

# Function to read Parquet file from ADLS
def read_from_adls(file_name):
    item_name = file_name + ".parquet"
    file_path = f"abfss://{produced_container_name}@{storage_account_name}.dfs.core.windows.net/{source_store_directory}/{item_name}"
    df = spark.read.parquet(file_path)
    return df

# Read the datasets back
datasets_to_read = [
    "X_scorched_particles_train_res",
    "X_scorched_particles_test_res",
    "y_scorched_particles_train_res",
    "y_scorched_particles_test_res"
]

# Dictionary to store the datasets
datasets = {}

for dataset_name in datasets_to_read:
    df = read_from_adls(dataset_name)
    datasets[dataset_name] = df.toPandas()

# Extract the datasets
X_scorched_particles_train = datasets["X_scorched_particles_train_res"]
X_scorched_particles_test = datasets["X_scorched_particles_test_res"]
y_scorched_particles_train = datasets["y_scorched_particles_train_res"]
y_scorched_particles_test = datasets["y_scorched_particles_test_res"]

# COMMAND ----------

# import the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
RANDOM_STATE = 123


# COMMAND ----------

# Define the feature names
feature_names = X_scorched_particles_train.columns

# Define plot feature importance function
def plot_feature_importance(importance, feature_names, title):
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.gca().invert_yaxis()
    plt.show()

# COMMAND ----------

# Define evaluate model function
def evaluate_model(dt_classifier):

    print("Train Accuracy : {0:.3f}".format(
        accuracy_score(y_scorched_particles_train, dt_classifier.predict(X_scorched_particles_train))))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_scorched_particles_train, dt_classifier.predict(X_scorched_particles_train)))
    print(classification_report(y_scorched_particles_train, dt_classifier.predict(X_scorched_particles_train)))

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_scorched_particles_train, dt_classifier.predict(X_scorched_particles_train))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    print("-"*50)
    print("Test Accuracy : {0:.3f}".format(
        accuracy_score(y_scorched_particles_test, dt_classifier.predict(X_scorched_particles_test))))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_scorched_particles_test, dt_classifier.predict(X_scorched_particles_test)))
    print(classification_report(y_scorched_particles_test, dt_classifier.predict(X_scorched_particles_test)))

    false_positive_rate_test, true_positive_rate_test, thresholds_test = roc_curve(
        y_scorched_particles_test, dt_classifier.predict(X_scorched_particles_test))
    roc_auc_test = auc(false_positive_rate_test, true_positive_rate_test)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate_test, true_positive_rate_test, 'b',
             label='AUC = %0.2f' % roc_auc_test)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build model
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Default Decison Tree

# COMMAND ----------

# Define the default DecisionTreeClassifier with a fixed random state
DT_default = DecisionTreeClassifier(random_state= RANDOM_STATE)

# Fit the decision tree classifier on the training data
DT_default.fit(X_scorched_particles_train, y_scorched_particles_train)

# Predict the labels for the test set
y_pred = DT_default.predict(X_scorched_particles_test)

# Calculate the accuracy of the model on the test set
DT_default_result = accuracy_score(y_scorched_particles_test, y_pred)

# Print the accuracy of the model on the test set
print('Accuracy on the test set:  {:.3f}'.format(DT_default_result))

# Print the depth of the decision tree
print('Depth of the decision tree: ', DT_default.get_depth())

# Print the number of leaves (terminal nodes) in the decision tree
print('Nodes of the decision tree: ', DT_default.get_n_leaves())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set max_depth

# COMMAND ----------

# Define the DecisionTreeClassifier with a maximum depth and a fixed random state
DT_default = DecisionTreeClassifier(max_depth=20, random_state= RANDOM_STATE)

# Fit the decision tree classifier on the training data
DT_default.fit(X_scorched_particles_train, y_scorched_particles_train)

# Predict the labels for the test set
y_pred = DT_default.predict(X_scorched_particles_test)

# Calculate the accuracy of the model on the test set
DT_default_result = accuracy_score(y_scorched_particles_test, y_pred)

# Print the accuracy of the model on the test set
print('Accuracy on the test set:  {:.3f}'.format(DT_default_result))

# Print the depth of the decision tree
print('Depth of the decision tree: ', DT_default.get_depth())

# Print the number of leaves (terminal nodes) in the decision tree
print('Nodes of the decision tree: ', DT_default.get_n_leaves())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross Validation

# COMMAND ----------

# Define the classifier with a maximum depth and a fixed random state
DT = DecisionTreeClassifier(max_depth=20, random_state= RANDOM_STATE)

# Fit the decision tree classifier on the training data
DT.fit(X_scorched_particles_train, y_scorched_particles_train)

# Perform cross-validation with 10 folds and use all available CPU cores for parallel processing
scores = cross_val_score(
    DT, 
    X_scorched_particles_train, 
    y_scorched_particles_train, 
    cv=10, 
    n_jobs=-1
)

# Print the cross-validation accuracy scores for each fold
print('Cross Validation accuracy scores: %s' % scores)

# Print the mean and standard deviation of the cross-validation accuracy scores
print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# COMMAND ----------

# MAGIC %md
# MAGIC  The cross validation accuracy scores for each of the 10 folds in the cross-validation process. Each value represents the accuracy of the model on one of the folds.
# MAGIC
# MAGIC  The cross-validation results by providing the mean accuracy and the standard deviation of the accuracy scores across all folds
# MAGIC

# COMMAND ----------

# Evaluate the decision tree model with CV
evaluate_model(DT)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter Tuning

# COMMAND ----------

# Define the Decision Tree model with hyperparameter tuning
dt = DecisionTreeClassifier(random_state= RANDOM_STATE)

# Define the expanded parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [None, 10, 20, 30, 50],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Perform hyperparameter tuning using GridSearchCV with cross-validation
dt_grid = GridSearchCV(
    dt, 
    param_grid, 
    cv=10, 
    scoring='accuracy'
)

dt_grid.fit(X_scorched_particles_train, y_scorched_particles_train)

# Output the best parameters
print("Best parameters for Decision Tree:", dt_grid.best_params_)

# Predict using the best model
y_test_pred_dt = dt_grid.predict(X_scorched_particles_test)

# COMMAND ----------

# Assign the best estimator (the best Decision Tree model found by GridSearchCV) to the variable 'dt'
dt = dt_grid.best_estimator_

# Evaluate the model 
evaluate_model(dt)

# COMMAND ----------

# Calculate and store feature importance values
dt_importance = dt_grid.best_estimator_.feature_importances_

plot_feature_importance(dt_importance, feature_names, "Decision Tree Feature Importance (Sorted) HT")

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize the structure of the best Decision Tree model

# COMMAND ----------

fig = plt.figure(figsize=(300,60))
_ = tree.plot_tree(
    dt_grid.best_estimator_,
    feature_names=X_scorched_particles_train.columns.tolist(),
    class_names=['0', '1'],
    fontsize=6,
    filled=True
)

# COMMAND ----------

# Export the decision tree structure as a textual representation
text_representation = tree.export_text(
    dt,
    feature_names=X_scorched_particles_train.columns.tolist()
)

# Print the textual representation of the decision tree
print(text_representation)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the Best Model to models folder

# COMMAND ----------

import joblib
import os

# Save the trained model to the specified directory
model_filename = "/Workspace/Repos/oce/michael.sun.onesource.APAC-OCE-db/src/ds_projects/prospecs/models/best_decision_tree_model_scorched_particles.pkl"
joblib.dump(dt_grid.best_estimator_, model_filename)