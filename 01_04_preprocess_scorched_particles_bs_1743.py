# Databricks notebook source
# MAGIC %md
# MAGIC ## Research Question: What are the optimal process specifications to satisfy optimal physical powder properties (PPP) for the SKU YCF325?

# COMMAND ----------

# MAGIC %md
# MAGIC ## This notebook is used for extracting and preparing the dataset for modeling PPP Scorched Particles (BS 1743).

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
# MAGIC ### Step 2: Filter the data to focus on a specific subset of interest, in this case, rows where TEST_FULL_CODE is ‘Scorched Particles (BS 1743)’.

# COMMAND ----------

# Create a temporary view
snowflake_table.createOrReplaceTempView("snowflake_table")

# Define the SQL query to select rows with SKU code 10449526 and the test full code 'Scorched Particles (BS 1743)'
query = """
SELECT * 
FROM snowflake_table
WHERE SKU_NUMBER = '10449526' AND TEST_FULL_CODE IN ('Scorched Particles (BS 1743)')
"""

# Execute the query and convert the result to a Pandas DataFrame
df = spark.sql(query).toPandas()

# COMMAND ----------

# Import the required libraries
import pandas as pd
import seaborn as sns
import pyspark.pandas as pspd
import matplotlib.pyplot as plt

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
    "SAP_BATCH_NUMBER",
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
    "PRODUCTION_STAGE"
]

# Remove the columns
filtered_df = df.drop(columns=columns_to_remove)

# Display the filtered DataFrame
display(filtered_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC The above columns that were removed were confirmed with D&I Vincent, as those variables are not useful for the project.

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
# MAGIC  Remove the rows with null values for DRYER_TIME_ON_PRODUCT_SINCE_CIP, PROCESS_BULK_DENSITY, PACKING_BULK_DENSITY, and POWDER_APPEARANCE_TEST_METHOD columns as confirmed with Vincent, because we still interested in the relationship between those variables and target value.

# COMMAND ----------

# Remove rows with null values in the specified columns
filtered_df = filtered_df.dropna(subset=[
    'DRYER_TIME_ON_PRODUCT_SINCE_CIP',
    'PROCESS_BULK_DENSITY',
    'PACKING_BULK_DENSITY',
    'POWDER_APPEARANCE_TEST_METHOD'
])

# Verify the changes
print(filtered_df.info())

# COMMAND ----------

# MAGIC %md
# MAGIC Check the unique values in each remaining column.

# COMMAND ----------

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


# COMMAND ----------

# MAGIC %md
# MAGIC These columns have only one unique value, indicating that they remain constant across all observations. As a result, they do not contribute to variability in the data set and are therefore uninformative for linear correlation or predictive modeling. We can remove these columns from the data set to reduce dimensionality and improve model performance.

# COMMAND ----------

columns_to_remove = [
    'SKU_NUMBER', 'CONCENTRATE_FILTER_CHANGE', 'FINES_A_DESTINATION',
    'FINES_B_DESTINATION', 'BLOWER_A_SPEED', 'BLOWER_B_SPEED',
    'POWDER_TRANSPORT_BREAKDOWN', 'LANCE_PITCH_INNER_LANCES',
    'LANCE_PITCH_OUTER_LANCES', 'TEST_FULL_CODE'
]

# Remove the columns from filtered_df
cleaned_df = filtered_df.drop(columns=columns_to_remove)

# Display the updated DataFrame
display(cleaned_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Handle Non-Numeric Data

# COMMAND ----------

# Display the categrical value counts for the column that contain string values
print(cleaned_df['POWDER_APPEARANCE_TEST_METHOD'].value_counts())
print(cleaned_df['BATCH_MILK_SOURCE'].value_counts())
print(cleaned_df['NOZZLE_SWIRL_INNER_LANCES'].value_counts())
print(cleaned_df['NOZZLE_SWIRL_OUTER_LANCES'].value_counts())


# COMMAND ----------

# Map the values for each specified column to convert categorical string values into numerical codes
cleaned_df['POWDER_APPEARANCE_TEST_METHOD'] = cleaned_df['POWDER_APPEARANCE_TEST_METHOD'].map({'Da Vinci': 0, 'Original': 1})
cleaned_df['NOZZLE_SWIRL_INNER_LANCES'] = cleaned_df['NOZZLE_SWIRL_INNER_LANCES'].map({'SW6': 0, 'SW5': 1})
cleaned_df['NOZZLE_SWIRL_OUTER_LANCES'] = cleaned_df['NOZZLE_SWIRL_OUTER_LANCES'].map({'SW6': 0, 'SW5': 1})

# COMMAND ----------

# MAGIC %md
# MAGIC The BATCH_MILK_SOURCE column contains strip leading or trailing spaces, we use following step to check and map values.

# COMMAND ----------

# Inspect unique values in the 'BATCH_MILK_SOURCE' column
unique_values = cleaned_df['BATCH_MILK_SOURCE'].unique()
print("Unique values before cleaning:", unique_values)

# Strip leading and trailing spaces and remove any hidden characters
cleaned_df['BATCH_MILK_SOURCE'] = cleaned_df['BATCH_MILK_SOURCE'].str.strip().str.replace(r'\s+', ' ', regex=True)

# Inspect unique values again after cleaning
unique_values_cleaned = cleaned_df['BATCH_MILK_SOURCE'].unique()
print("Unique values after cleaning:", unique_values_cleaned)

# Map the values to numerical codes
cleaned_df['BATCH_MILK_SOURCE'] = cleaned_df['BATCH_MILK_SOURCE'].map({
    '100% liquid milk': 0,
    '100% powder': 1,
    '50% liquid milk + 50% powder': 2
})

# Check for any remaining NaN values
nan_values = cleaned_df[cleaned_df['BATCH_MILK_SOURCE'].isna()]
print("Rows with NaN values after mapping:", nan_values)

# COMMAND ----------

# Check the value mapping is correct or not
print(cleaned_df['BATCH_MILK_SOURCE'].value_counts())

# COMMAND ----------

# Set the updated DataFrame as a new DataFrame
new_df = cleaned_df

# Display the new DataFrame
display(new_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the Scorched Particles (BS 1743) actual result to binary. Set A to 1, meaning in_spec, and set other values to 0, meaning oo_spec, for future data modeling.

# COMMAND ----------

print(new_df['TEST_ACTUAL_RESULT'].value_counts())

# COMMAND ----------

# Filter out rows where 'TEST_ACTUAL_RESULT' column has the value '#'
new_df = new_df[new_df['TEST_ACTUAL_RESULT'] != '#']

# Check the distribution of the 'TEST_ACTUAL_RESULT' column
distribution = new_df['TEST_ACTUAL_RESULT'].value_counts()

# Display the distribution
display(distribution)

# COMMAND ----------

# Set 'A' as 1 and other values as 0 in the 'TEST_ACTUAL_RESULT' column
new_df['TEST_ACTUAL_RESULT'] = new_df['TEST_ACTUAL_RESULT'].apply(lambda x: 1 if x == 'A' else 0)

# Check the distribution of the 'TEST_ACTUAL_RESULT' column
distribution = new_df['TEST_ACTUAL_RESULT'].value_counts()

# Display the distribution
print(distribution)


# COMMAND ----------

# Final check the cleaned dataset
print(new_df.info())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Data spliting

# COMMAND ----------

# MAGIC %md
# MAGIC Split the dataset into training and validation sets. This split is essential for training machine learning models, as it allows us to evaluate the model’s performance on the validation set.

# COMMAND ----------

# Data spliting
from sklearn.model_selection import train_test_split

X = new_df.drop(columns=['TEST_ACTUAL_RESULT'])
y = new_df['TEST_ACTUAL_RESULT']

# Split the data into train+test and validation sets
X_scorched_particles_bs_1743, X_scorched_particles_bs_1743_val, y_scorched_particles_bs_1743, y_scorched_particles_bs_1743_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nX_scorched_particles_bs_1743:")
print(X_scorched_particles_bs_1743)
print("\ny__scorched_particles_bs_1743:")
print(y_scorched_particles_bs_1743)
print("\nX__scorched_particles_bs_1743_val:")
print(X_scorched_particles_bs_1743_val)
print("\ny__scorched_particles_bs_1743_val")
print(y_scorched_particles_bs_1743_val)

# COMMAND ----------

# Check and convert y_scorched_particles_bs_1743, y_scorched_particles_bs_1743_val to pandas DataFrame 
datasets_to_check = {
    "y_scorched_particles_bs_1743": y_scorched_particles_bs_1743,
    "y_scorched_particles_bs_1743_val": y_scorched_particles_bs_1743_val,
}

for name, dataset in datasets_to_check.items():
    if not isinstance(dataset, pd.DataFrame):
        datasets_to_check[name] = pd.DataFrame(dataset, columns=['TEST_ACTUAL_RESULT'])

# Update the variables with the converted DataFrames
y_scorched_particles_bs_1743 = datasets_to_check["y_scorched_particles_bs_1743"]
y_scorched_particles_bs_1743_val = datasets_to_check["y_scorched_particles_bs_1743_val"]

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

# Convert pandas DataFrames to Spark DataFrames and save them
datasets = {
    "X_scorched_particles_bs_1743": X_scorched_particles_bs_1743,
    "X_scorched_particles_bs_1743_val": X_scorched_particles_bs_1743_val,
    "y_scorched_particles_bs_1743": y_scorched_particles_bs_1743,
    "y_scorched_particles_bs_1743_val": y_scorched_particles_bs_1743_val
}

for name, dataset in datasets.items():
    if isinstance(dataset, pd.DataFrame):
        spark_df = spark.createDataFrame(dataset)
    else:
        raise ValueError(f"Dataset {name} is not a pandas DataFrame")
    display(spark_df)
    save_to_adls(spark_df, name)