# Databricks notebook source
# MAGIC %md
# MAGIC ## Research Question: What are the optimal process specifications to satisfy optimal physical powder properties (PPP) for the SKU YCF325?

# COMMAND ----------

# MAGIC %md
# MAGIC ## This notebook is used for extracting and preparing the dataset for modeling PPP Foam Height (DV).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Extract the dataset table from Snowflake.

# COMMAND ----------

# Connect to Snowflake table
snowflake_table = (spark.read
  .format("snowflake")
  .option("dbtable", "F_RNI_BAL_PO_PROCESS_READINGS_AND_LAB_RESULTS")
  .option("sfUrl", "danonesea.southeast-asia.azure.snowflakecomputing.com")
  .option("sfUser", dbutils.secrets.get(scope="DAN-AP-T-KVT800-R-OCE-DB", key="snowflake-user"))
  .option("sfPassword", dbutils.secrets.get(scope="DAN-AP-T-KVT800-R-OCE-DB", key="snowflake-password"))
  .option("sfDatabase", "DEV_OCE")
  .option("sfSchema", "OCE_DSP")
  .option("sfWarehouse", "DEV_OCE_ELT_WH")
  .load()
)
display(snowflake_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Filter the data to focus on a specific subset of interest, in this case, rows where TEST_FULL_CODE is ‘Foam Height (DV)’.

# COMMAND ----------

# Create a temporary view
snowflake_table.createOrReplaceTempView("snowflake_table")

# Define the SQL query to select rows with SKU code 10449526
query = """
SELECT * 
FROM snowflake_table
WHERE SKU_NUMBER = '10449526' AND TEST_FULL_CODE IN ('Foam Height (DV)')
"""

# Execute the query and convert the result to a Pandas DataFrame
df = spark.sql(query).toPandas()

# COMMAND ----------

# Import the required libraries
import pandas as pd
import seaborn as sns
import pyspark.pandas as pspd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Check for missing values and NA values in the dataset.

# COMMAND ----------

# Identify columns with NA values
columns_with_na = df.columns[df.isna().any()].tolist()

# Count how many NA values each column contains
na_count = df.isna().sum()

# Create a DataFrame of columns that have NA values and their corresponding NA counts
columns_with_na_count = pd.DataFrame({
    'NA_Count': na_count[columns_with_na]
})

# Display the result as a table
print(columns_with_na_count)


# COMMAND ----------

# MAGIC %md
# MAGIC ###  Step 4: Remove columns with NA (missing values) and columns that are not part of the processing parameters.

# COMMAND ----------

# List of columns to remove
columns_to_remove = [
    "DATETIME_UTC",
    "PO_NUMBER",
    "SAP_BATCH_NUMBER",
    "SKU_NUMBER",
    "PARAMETERS_VERSION_NUMBER",
    "RUN_BATCH_NUMBER",
    "BATCH_MILK_SOURCE",
    "DRYER_TIME_ON_HEAT_SINCE_CIP",
    "DRYER_TIME_ON_WATER_SINCE_CIP",
    "NIR_POWDER_MOISTURE",
    "NIR_POWDER_FAT",
    "NIR_POWDER_PROTEIN",
    "SEDIMENT_PAD_READING",
    "POWDER_APPEARANCE_INITIAL_READING",
    "POWDER_APPEARANCE_RETEST",
    "DA_VINCI_INITIAL_TEST_YELLOW_COUNT",
    "DA_VINCI_INITIAL_TEST_BROWN_COUNT",
    "DA_VINCI_INITIAL_TEST_BLACK_COUNT",
    "DA_VINCI_RETEST_TEST_YELLOW_COUNT",
    "DA_VINCI_RETEST_TEST_BROWN_COUNT",
    "DA_VINCI_RETEST_TEST_BLACK_COUNT",
    "LANCE_ATOMISER_RING_SETUP",
    "TEST_UOM",
    "TEST_TARGET",
    "TEST_MINIMUM_LIMIT",
    "TEST_MAXIMUM_LIMIT",
    "NOZZLE_SETUPS_VERSION_NUMBER",
    "TEST_SHORT_CODE",
    "TEST_FULL_DESCRIPTION",
    "SAMPLE_DESCRIPTION",
    "PRODUCTION_STAGE",
    "TEST_FULL_CODE",
    "PROCESS_BULK_DENSITY",
    "PACKING_BULK_DENSITY",
    "POWDER_APPEARANCE_TEST_METHOD",
    "REWORK_ADDED_QUANTITY",
    "EFB_SECTION_1_AIR_PRESSURE",
    "EFB_SECTION_2_AIR_PRESSURE",
    "EFB_SECTION_3_AIR_PRESSURE",
    "EFB_1_AIR_TEMPERATURE_TARGET",
    "EFB_2_AIR_TEMPERATURE_TARGET",
    "EFB_3_AIR_TEMPERATURE_TARGET"
]

# Remove the columns
filtered_df = df.drop(columns=columns_to_remove)

# COMMAND ----------

# MAGIC %md
# MAGIC The above columns that were removed were confirmed with D&I Vincent, as those variables are not useful for the project.
# MAGIC "EFB_SECTION_1_AIR_PRESSURE", "EFB_SECTION_2_AIR_PRESSURE", "EFB_SECTION_3_AIR_PRESSURE", "EFB_1_AIR_TEMPERATURE_TARGET", "EFB_2_AIR_TEMPERATURE_TARGET", "EFB_3_AIR_TEMPERATURE_TARGET" variables removed due to data confudion.

# COMMAND ----------

# MAGIC %md
# MAGIC Then, we check for missing values in the remaining columns.

# COMMAND ----------

# Count missing values for each column
null_counts = filtered_df.isna().sum()

# Filter columns with null values
null_columns = null_counts[null_counts > 0]

# Print columns with their null value counts
print(null_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC Remove the rows with null values in the specified columns, because we still interested in the relationship between those variables and target value. And check again for null vaule.

# COMMAND ----------

# Remove rows with null values in the specified columns
filtered_df = filtered_df.dropna()

# Count missing values for each column
null_counts = filtered_df.isna().sum()

# Filter columns with null values
null_columns = null_counts[null_counts > 0]

# Print columns with their null value counts again
print(null_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC Check the unique values in each remaining column.

# COMMAND ----------

# Check the unique values in each remaining column. and remove that column if there is a unique value in that column, then check again for unique value column

#  Check the unique values in filtered_df
unique_values = filtered_df.nunique()
columns_with_one_unique_value = unique_values[unique_values == 1].index

# Create a DataFrame with columns that have only one unique value
columns_with_one_unique_value_df = pd.DataFrame({
    'Column_Name': columns_with_one_unique_value,
    'Unique_Value': [filtered_df[col].unique()[0] for col in columns_with_one_unique_value]
})

# Display the DataFrame
print(columns_with_one_unique_value_df)


# Remove columns with only one unique value
filtered_df = filtered_df.drop(columns=columns_with_one_unique_value)

# Verify if there are any columns left with only one unique value after removal
unique_values_after_removal = filtered_df.nunique()
columns_with_one_unique_value_after_removal = unique_values_after_removal[unique_values_after_removal == 1].index

# Print the remaining columns with only one unique value (if any)
if not columns_with_one_unique_value_after_removal.empty:
    columns_with_one_unique_value_df_after_removal = pd.DataFrame({
        'Column_Name': columns_with_one_unique_value_after_removal,
        'Unique_Value': [filtered_df[col].unique()[0] for col in columns_with_one_unique_value_after_removal]
    })
    print("Columns with one unique value after removal:")
    print(columns_with_one_unique_value_df_after_removal)

    # Remove the remaining columns with one unique value (if any)
    filtered_df = filtered_df.drop(columns=columns_with_one_unique_value_after_removal)

    # Print the final DataFrame shape
    print("\nShape of the DataFrame after removing columns with one unique value:", filtered_df.shape)
else:
    print("\nNo columns with only one unique value found after the initial removal.")
    print("Shape of the DataFrame:", filtered_df.shape)


# COMMAND ----------

# MAGIC %md
# MAGIC These columns have only one unique value, indicating that they remain constant across all observations. As a result, they do not contribute to variability in the data set and are therefore uninformative for linear correlation or predictive modeling. We can remove these columns from the data set to reduce dimensionality and improve model performance.

# COMMAND ----------

cleaned_df = filtered_df
# Display the updated DataFrame
display(cleaned_df.info())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Handle Non-Numeric Data

# COMMAND ----------

# Display the categrical value counts for the column that contain string values
print(cleaned_df['NOZZLE_SWIRL_INNER_LANCES'].value_counts())
print(cleaned_df['NOZZLE_SWIRL_OUTER_LANCES'].value_counts())


# COMMAND ----------

cleaned_df['NOZZLE_SWIRL_INNER_LANCES'] = cleaned_df['NOZZLE_SWIRL_INNER_LANCES'].map({'SW6': 0, 'SW5': 1})
cleaned_df['NOZZLE_SWIRL_OUTER_LANCES'] = cleaned_df['NOZZLE_SWIRL_OUTER_LANCES'].map({'SW6': 0, 'SW5': 1})

# COMMAND ----------

# Set the updated DataFrame as a new DataFrame
new_df = cleaned_df

# Display the new DataFrame
display(new_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the form height actual result to binary. Set 10 and 20 to 1, meaning in_spec, and set rest of values (30, 40, 50) to 0, meaning oo_spec, for future data modeling.

# COMMAND ----------

print(new_df['TEST_ACTUAL_RESULT'].value_counts())

# COMMAND ----------

# Convert the 'TEST_ACTUAL_RESULT' column to integers
new_df['TEST_ACTUAL_RESULT'] = new_df['TEST_ACTUAL_RESULT'].astype(int)

# Set 10 and 20 as 1 and the rest of the numbers as 0 in the 'TEST_ACTUAL_RESULT' column
new_df['TEST_ACTUAL_RESULT'] = new_df['TEST_ACTUAL_RESULT'].apply(lambda x: 1 if x in [10, 20] else 0)

# Check the distribution of the 'TEST_ACTUAL_RESULT' column
distribution = new_df['TEST_ACTUAL_RESULT'].value_counts()

# Display the distribution
print(distribution)

# COMMAND ----------

# Check unique count in each columns in new_df, if the unque count less than 5 change the data type to categrical, greater than 5, change the data type to numrical type

# Check unique counts and adjust data types accordingly
for col in new_df.columns:
  unique_count = new_df[col].nunique()
  if unique_count < 5:
    new_df[col] = new_df[col].astype('category')
  else:
    # Attempt to convert to numerical, handle errors gracefully
    try:
      new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    except:
      pass # Leave as is if conversion fails

# Final check the cleaned dataset
print(new_df.info())

# COMMAND ----------

# MAGIC %md
# MAGIC Correlation check

# COMMAND ----------


correlation_matrix = new_df.corr()

# Display the correlation matrix using a heatmap
plt.figure(figsize=(60, 60)) 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
fontsize=6,
plt.show()

# COMMAND ----------

import numpy as np
import pandas as pd

def remove_highly_correlated_features(df, threshold=0.8):
    """
    Removes one of the highly correlated features from a pandas DataFrame,
    printing highly correlated pairs and dropped columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): The correlation threshold above which to remove features.

    Returns:
        pd.DataFrame: The DataFrame with highly correlated features removed.
    """

    # Calculate the correlation matrix
    correlation_matrix = df.corr().abs()

    # Create a mask for the upper triangle of the correlation matrix
    upper_triangle_mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)

    # Find the columns to drop and store correlated pairs
    columns_to_drop = []
    correlated_pairs = []  # Store correlated pairs

    for column in correlation_matrix.columns:
        # Get the correlations for the current column
        correlations = correlation_matrix[column]

        # Filter correlations based on the upper triangle mask and threshold
        filtered_correlations = correlations[upper_triangle_mask[correlation_matrix.columns.get_loc(column)]]  
        highly_correlated_columns = filtered_correlations[filtered_correlations > threshold].index

        # Add the first highly correlated column to the list of columns to drop
        # and store the correlated pair
        for correlated_column in highly_correlated_columns:
            if correlated_column not in columns_to_drop:
                columns_to_drop.append(correlated_column)
                correlated_pairs.append((column, correlated_column, correlations[correlated_column]))  
                break  # Break to avoid adding multiple pairs for the same column

    # Print the highly correlated pairs
    print("Highly Correlated Pairs (above threshold {:.2f}):".format(threshold))
    for pair in correlated_pairs:
        print(f"{pair[0]} and {pair[1]} (Correlation: {pair[2]:.2f})")

    # Print the columns to be dropped as a table
    print("\nDropping the following columns due to high correlation:")
    if columns_to_drop:
        columns_to_drop_df = pd.DataFrame(columns_to_drop, columns=['Dropped Columns'])
        display(columns_to_drop_df)
    else:
        print("No columns dropped due to high correlation.")

    # Drop the selected columns from the DataFrame
    df = df.drop(columns=columns_to_drop)

    return df

# COMMAND ----------

# Removw high correlated feature
new_df = remove_highly_correlated_features(new_df)

# COMMAND ----------

# Convert 'TEST_ACTUAL_RESULT' to numerical type
new_df['TEST_ACTUAL_RESULT'] = new_df['TEST_ACTUAL_RESULT'].astype('int')

# Recalculate the correlation matrix based on the current new_df
correlation_matrix = new_df.corr()

# Identify columns to drop based on correlation with 'TEST_ACTUAL_RESULT'
columns_to_drop = correlation_matrix[
    (correlation_matrix['TEST_ACTUAL_RESULT'] < 0.03) & 
    (correlation_matrix['TEST_ACTUAL_RESULT'] > -0.03)
].index

# Display the columns that have been dropped
print("Columns dropped due to low correlation with 'TEST_ACTUAL_RESULT':")
print(columns_to_drop)

# Drop the identified columns
new_df = new_df.drop(columns=columns_to_drop)

# Update the correlation matrix (again, for consistency)
correlation_matrix = new_df.corr()

# Display the updated correlation matrix
display(new_df.corr('pearson')['TEST_ACTUAL_RESULT'].sort_values(ascending=False))

# COMMAND ----------

new_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Data splitting

# COMMAND ----------

# MAGIC %md
# MAGIC Split the dataset into training and validation sets. This split is essential for training machine learning models, as it allows us to evaluate the model’s performance on the validation set.

# COMMAND ----------

# MAGIC %md
# MAGIC Balance sample

# COMMAND ----------

# Install the imblearn library
%pip install imbalanced-learn

# COMMAND ----------

# Data splitting
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Split the data into train+test and validation sets
X = new_df.drop(columns=['TEST_ACTUAL_RESULT'])
y = new_df['TEST_ACTUAL_RESULT']

X_foam_height_train, X_foam_height_temp, y_foam_height_train, y_foam_height_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Split temporary set into test and validation sets
X_foam_height_test, X_foam_height_val, y_foam_height_test, y_foam_height_val = train_test_split(
    X_foam_height_temp, y_foam_height_temp, test_size=0.5, random_state=42
)  # Split temp into 50% test, 50% validation

# Apply SMOTE only to the training and test sets
oversample = SMOTE()

X_foam_height_train = X_foam_height_train.apply(pd.to_numeric, errors='coerce')
y_foam_height_train = pd.to_numeric(y_foam_height_train, errors='coerce')

X_foam_height_test = X_foam_height_test.apply(pd.to_numeric, errors='coerce')
y_foam_height_test = pd.to_numeric(y_foam_height_test, errors='coerce')

# Print counts before resampling
print("Before OverSampling, counts of label '1' in y train: {}".format(sum(y_foam_height_train == 1)))
print("Before OverSampling, counts of label '0' in y train: {} \n".format(sum(y_foam_height_train == 0)))

print("Before OverSampling, counts of label '1' in y test: {}".format(sum(y_foam_height_test == 1)))
print("Before OverSampling, counts of label '0' in y test: {} \n".format(sum(y_foam_height_test == 0)))

X_foam_height_train_res, y_foam_height_train_res = oversample.fit_resample(X_foam_height_train, y_foam_height_train)
X_foam_height_test_res, y_foam_height_test_res = oversample.fit_resample(X_foam_height_test, y_foam_height_test)

print('After OverSampling, the shape of X_train_res: {}'.format(X_foam_height_train_res.shape))
print('After OverSampling, the shape of y_train_res: {} \n'.format(y_foam_height_train_res.shape))

print('After OverSampling, the shape of X_test_res: {}'.format(X_foam_height_test_res.shape))
print('After OverSampling, the shape of y_test_res: {} \n'.format(y_foam_height_test_res.shape))

print("After OverSampling, counts of label '1' in y train_res: {}".format(sum(y_foam_height_train_res == 1)))
print("After OverSampling, counts of label '0' in y train_res: {}".format(sum(y_foam_height_train_res == 0)))

print("After OverSampling, counts of label '1' in y test_res: {}".format(sum(y_foam_height_test_res == 1)))
print("After OverSampling, counts of label '0' in y test_res: {}".format(sum(y_foam_height_test_res == 0)))

# Check the count of the validation dataset
print("Validation dataset shape: {}".format(X_foam_height_val.shape))
print("Counts of label '1' in validation: {}".format(sum(y_foam_height_val == 1)))
print("Counts of label '0' in validation: {}".format(sum(y_foam_height_val == 0)))

# COMMAND ----------

# Check and convert y_foam_height, y_foam_height_val to pandas DataFrame 
datasets_to_check = {
    "y_foam_height_train_res": y_foam_height_train_res,
    "y_foam_height_test_res": y_foam_height_test_res,
    "y_foam_height_val": y_foam_height_val,
}

for name, dataset in datasets_to_check.items():
    if not isinstance(dataset, pd.DataFrame):
        datasets_to_check[name] = pd.DataFrame(dataset, columns=['TEST_ACTUAL_RESULT'])

# Update the variables with the converted DataFrames
y_foam_height_train_res = datasets_to_check["y_foam_height_train_res"]
y_foam_height_test_res = datasets_to_check["y_foam_height_test_res"]
y_foam_height_val = datasets_to_check["y_foam_height_val"]

# COMMAND ----------

import datetime

"""Import parameters from Azure Data Factory analytical pipeline
These parameters are used in the Extractor class methods and for
connection to Azure Data Lake Storage storage containers"""

application_id = dbutils.secrets.get(scope="DAN-AP-T-KVT800-R-OCE-DB", key="databricks-dev-application-id")
scope_name = dbutils.secrets.get(scope="DAN-AP-T-KVT800-R-OCE-DB", key="databricks-dev-scope-name")
service_credentials_key_name = dbutils.secrets.get(scope="DAN-AP-T-KVT800-R-OCE-DB", key="databricks-dev-service-credentials-key-name")
storage_account_name = dbutils.secrets.get(scope="DAN-AP-T-KVT800-R-OCE-DB", key="databricks-dev-storage-account-name")
source_store_directory = dbutils.secrets.get(scope="DAN-AP-T-KVT800-R-OCE-DB", key="databricks-dev-source-store-directory")

# Setup parameters for connection to Azure Data Lake Storage storage containers
produced_container_name = "produced"
spark.conf.set("fs.azure.account.auth.type", "OAuth")
spark.conf.set(
    "fs.azure.account.oauth.provider.type",
    "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
)
spark.conf.set(
    "fs.azure.account.oauth2.client.endpoint",
    "https://login.microsoftonline.com/4720ed5e-c545-46eb-99a5-958dd333e9f2/oauth2/token",
)
spark.conf.set("fs.azure.account.oauth2.client.id", application_id)
spark.conf.set(
    "fs.azure.account.oauth2.client.secret",
    dbutils.secrets.get(scope=scope_name, key=service_credentials_key_name),
)

# Function to save DataFrame to ADLS
def save_to_adls(df, file_name):
    item_name = file_name + ".parquet"
    produced_file_path = f"abfss://{produced_container_name}@{storage_account_name}.dfs.core.windows.net/{source_store_directory}/{item_name}"
    df.write.parquet(produced_file_path, mode='overwrite', compression='uncompressed')

datasets = {
    "X_foam_height_train_res": X_foam_height_train_res,
    "X_foam_height_test_res": X_foam_height_test_res,
    "X_foam_height_val": X_foam_height_val,

    "y_foam_height_train_res": y_foam_height_train_res,
    "y_foam_height_test_res": y_foam_height_test_res,
    "y_foam_height_val": y_foam_height_val
}

for name, dataset in datasets.items():
    if isinstance(dataset, pd.DataFrame):
        spark_df = spark.createDataFrame(dataset)
    else:
        raise ValueError(f"Dataset {name} is not a pandas DataFrame")
    display(spark_df)
    save_to_adls(spark_df, name)