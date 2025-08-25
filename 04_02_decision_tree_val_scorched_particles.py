# Databricks notebook source
# MAGIC %md
# MAGIC # Research Question: What are the optimal process specifications to satisfy optimal physical powder properties (PPP) for the SKU YCF325?

# COMMAND ----------

# MAGIC %md
# MAGIC ## This notebook is used for validating the best decision tree modelling for PPP Scorched Particles.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read validation Parquet files from ADLS into Pandas DataFrames for validating tasks.

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
    "X_scorched_particles_val",
    "y_scorched_particles_val"
]

# Dictionary to store the datasets
datasets = {}

for dataset_name in datasets_to_read:
    df = read_from_adls(dataset_name)
    datasets[dataset_name] = df.toPandas()

# Extract the datasets
X_scorched_particles_val = datasets["X_scorched_particles_val"]
y_scorched_particles_val = datasets["y_scorched_particles_val"]

# COMMAND ----------

# import the required libraries
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
RANDOM_STATE = 123


# COMMAND ----------

# Load the best model from the folder
model_path = "/Workspace/Repos/oce/michael.sun.onesource.APAC-OCE-db/src/ds_projects/prospecs/models/best_decision_tree_model_scorched_particles.pkl"
best_model = joblib.load(model_path)

# COMMAND ----------

# Predict using the best model on the validation set
y_val_pred = best_model.predict(X_scorched_particles_val)

# Evaluate performance on the validation set
print("Validation Accuracy:", accuracy_score(y_scorched_particles_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_scorched_particles_val, y_val_pred))
print("Classification Report:\n", classification_report(y_scorched_particles_val, y_val_pred))

# COMMAND ----------

# Calculate the ROC curve for the best model
y_val_prob_dt = best_model.predict_proba(X_scorched_particles_val)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_scorched_particles_val, y_val_prob_dt)

roc_auc_dt = auc(fpr_dt, tpr_dt)

# Plot the ROC curves for all models
plt.figure(figsize=(10, 6))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Validation Set)')  # Updated title
plt.legend(loc='lower right')
plt.show()