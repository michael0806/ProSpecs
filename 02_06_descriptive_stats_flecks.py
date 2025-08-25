# Databricks notebook source
# MAGIC %md
# MAGIC # Research Question: What are the optimal process specifications to satisfy optimal physical powder properties (PPP) for the SKU YCF325?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## This notebook is used for descriptive statistics of Flecks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract data from Snowflake.

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

# Create a temporary view
snowflake_table.createOrReplaceTempView("snowflake_table")

# Define the SQL query to select rows with SKU code 10449526
query = """
SELECT * 
FROM snowflake_table
WHERE SKU_NUMBER = '10449526' AND TEST_FULL_CODE IN ('Flecks')
"""

# Execute the query and convert the result to a Pandas DataFrame
df = spark.sql(query).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preproccess dataset

# COMMAND ----------

import pandas as pd

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

# COMMAND ----------

# Count missing values for each column
null_counts = filtered_df.isna().sum()

# Filter columns with null values
null_columns = null_counts[null_counts > 0]

# Print columns with their null value counts
print(null_columns)

# COMMAND ----------

# Remove rows with null values in the specified columns
filtered_df = filtered_df.dropna(subset=[
    'DRYER_TIME_ON_PRODUCT_SINCE_CIP',
    'PROCESS_BULK_DENSITY',
    'PACKING_BULK_DENSITY',
    'POWDER_APPEARANCE_TEST_METHOD'
])

# Verify the changes
display(filtered_df.info())

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the flecks actual result to binary. Set A+ and A- to 1 (in spec) and the remaining results to 0 (out of spec).

# COMMAND ----------

# Check the distribution of test actual result
print(filtered_df['TEST_ACTUAL_RESULT'].value_counts())

# COMMAND ----------

# Map 'A+' and 'A-' to 1, and other values to 0 in the 'TEST_ACTUAL_RESULT' column
filtered_df['TEST_ACTUAL_RESULT'] = filtered_df['TEST_ACTUAL_RESULT'].apply(lambda x: 1 if x in ['A+', 'A-'] else 0)

# Check the distribution of the 'TEST_ACTUAL_RESULT' column
distribution = filtered_df['TEST_ACTUAL_RESULT'].value_counts()

# Display the distribution
print(distribution)

# COMMAND ----------

# Check the preprocessing dataset
display(filtered_df.info())
display(filtered_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Descriptive statistics from independent variables and the relation with target variable.
# MAGIC =================

# COMMAND ----------


import matplotlib.pyplot as plt
import seaborn as sns

# Group by TEST_ACTUAL_RESULT and count the occurrences
result_summary = filtered_df.groupby('TEST_ACTUAL_RESULT').size().reset_index(name='Count')

# Calculate the total count for percentages
total_count = result_summary['Count'].sum()

# Set up the color palette
palette = sns.color_palette("husl", len(result_summary['TEST_ACTUAL_RESULT'].unique()))

# Create bar plot with labels
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='TEST_ACTUAL_RESULT', y='Count', data=result_summary, palette=palette)

# Add text labels on the bars
for index, row in result_summary.iterrows():
    count = int(row['Count'])
    percentage = (count / total_count) * 100
    bar_plot.text(index, row['Count'] + 0.05, f'{count} ({percentage:.1f}%)', ha='center', va='bottom')

# Add labels and title
plt.title("Count of Each TEST_ACTUAL_RESULT value for Flecks")
plt.xlabel("TEST_ACTUAL_RESULT")
plt.ylabel("Count")

# Show plot
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. RAW INPUTS & BATCH INFO   
# MAGIC For this step, we do descriptive statistic analysis for the raw input stage parameters. 
# MAGIC includes:
# MAGIC - PO_NUMBER
# MAGIC - RUN_BATCH_NUMBER
# MAGIC - BATCH_MILK_SOURCE 
# MAGIC
# MAGIC Then check the relationship betwweent those varibales and test actual result for Flecks.

# COMMAND ----------

def show_unique_values(df, variables):
    """
    Show variable name, unique count, unique values, and the count of each unique value for a given set of variables in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    variables (list): A list of variable names to show unique values for.
    
    Returns:
    pd.DataFrame: A DataFrame containing the variable name, unique count, unique values, and the count of each unique value.
    """
    unique_table = pd.DataFrame(columns=["Variable", "Unique_Count", "Unique_Values", "Value_Counts"])

    for var in variables:
        data = df[var]
        value_counts = data.value_counts().to_dict()
        result = {
            "Variable": var,
            "Unique_Count": data.nunique(),
            "Unique_Values": data.unique().tolist(),
            "Value_Counts": value_counts
        }
        unique_table = unique_table.append(result, ignore_index=True)

    return unique_table

# COMMAND ----------

# display the unique values for the batch variables
batch_vars = [
  "PO_NUMBER",
  "PARAMETERS_VERSION_NUMBER",
  "RUN_BATCH_NUMBER",
  "BATCH_MILK_SOURCE"
]

# Calculate the unique values for the batch variables
unique_table = show_unique_values(filtered_df, batch_vars)

# Display the table
display(unique_table)

# COMMAND ----------

# Group by PO_NUMBER and TEST_ACTUAL_RESULT and count the occurrences
result_summary = filtered_df.groupby(['PO_NUMBER', 'TEST_ACTUAL_RESULT']).size().reset_index(name='Count')

# Increase the figure size
plt.figure(figsize=(16, 10))

# Plot the data with a horizontal bar plot
bar_plot = sns.barplot(data=result_summary, y='PO_NUMBER', x='Count', hue='TEST_ACTUAL_RESULT', palette='husl', orient='h')

# Add text labels on the bars
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_width(), '.0f'), 
                      (p.get_width(), p.get_y() + p.get_height() / 2.), 
                      ha = 'center', va = 'center', 
                      xytext = (9, 0), 
                      textcoords = 'offset points')

plt.title('PO_NUMBER vs TEST_ACTUAL_RESULT with Counts')
plt.xlabel('Count')
plt.ylabel('PO_NUMBER')
plt.legend(title='TEST_ACTUAL_RESULT')
plt.grid(True)
plt.show()



# COMMAND ----------

def plot_categorical_summary(df, category1, category2, legend_loc='upper left'):
    """
    Create a bar plot for two categorical variables with counts and percentages on the bars.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    category1 (str): The first categorical variable.
    category2 (str): The second categorical variable.
    legend_loc (str): The location of the legend.
    """
    # Calculate counts and percentages
    summary = df.groupby([category1, category2]).size().reset_index(name='Count')
    total_counts = summary.groupby(category1)['Count'].transform('sum')
    summary['Percentage'] = (summary['Count'] / total_counts * 100).round(2)

    # Create a bar plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=summary, x=category2, y='Count', hue=category1, dodge=True)

    # Add percentages on the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.0f}\n({height / summary["Count"].sum() * 100:.1f}%)', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')

    # Set labels and title
    ax.set_xlabel(category2)
    ax.set_ylabel('Count')
    ax.set_title(f'{category1} vs {category2} with Counts and Percentage')
    plt.xticks(rotation=90)

    # Position the legend
    ax.legend(title=category1, loc=legend_loc)

    plt.show()


# COMMAND ----------

# PARAMETERS_VERSION_NUMBER VS TEST_ACTUAL_RESULT
plot_categorical_summary(filtered_df, 'PARAMETERS_VERSION_NUMBER', 'TEST_ACTUAL_RESULT')

# COMMAND ----------

# RUN_BATCH_NUMBER VS TEST_ACTUAL_RESULT
plot_categorical_summary(filtered_df, 'RUN_BATCH_NUMBER', 'TEST_ACTUAL_RESULT')

# COMMAND ----------

# BATCH_MILK_SOURCE VS TEST_ACTUAL_RESULT
plot_categorical_summary(filtered_df, 'BATCH_MILK_SOURCE', 'TEST_ACTUAL_RESULT')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. EVAPORATOR STAGE 

# COMMAND ----------

# MAGIC %md
# MAGIC For this step, we do descriptive statistic analysis for the evaporator stage processing parameters. 
# MAGIC includes: 
# MAGIC - EVAPORATOR_NUMBER_IN_USE
# MAGIC - EVAPORATOR_FEED_FLOW
# MAGIC - EVAPORATOR_EFFECT_1_HEATING_TEMPERATURE
# MAGIC - EVAPORATOR_EFFECT_1_BOILING_TEMPERATURE
# MAGIC - EVAPORATOR_EFFECT_1_TEMPERATURE_DIFFERENTIAL
# MAGIC - TVR_STEAM_PRESSURE
# MAGIC - EVAPORATOR_CONCENTRATE_CALC_TS
# MAGIC - EVAPORATOR_CONCENTRATE_FLOWRATE
# MAGIC - EVAPORATOR_CONCENTRATE_DENSITY
# MAGIC - EVAPORATOR_COOLING_WATER_TEMP
# MAGIC - FLASH_VESSEL_NUMBER
# MAGIC - SECOND_STAGE_DSI_TEMPERATURE
# MAGIC
# MAGIC Then analyze the relationship between each processing parameter and the target variable which is the Flecks actual test result. 

# COMMAND ----------

# List the variables of interest for the evaporator stage
evap_vars = [
  "EVAPORATOR_NUMBER_IN_USE",
]

# Calculate the unique values 
unique_table = show_unique_values(filtered_df, evap_vars)

# Display the table
display(unique_table)

# COMMAND ----------

plot_categorical_summary(filtered_df, 'EVAPORATOR_NUMBER_IN_USE', 'TEST_ACTUAL_RESULT')

# COMMAND ----------

def calculate_statistics(df, variables):
    """
    Calculate statistics for a given set of variables in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    variables (list): A list of variable names to calculate statistics for.
    
    Returns:
    pd.DataFrame: A DataFrame containing the calculated statistics.
    """
    results_table = pd.DataFrame(columns=["Variable", "Unique_Count", "Min", "Max", "Mean", "Median", "Mode", "SD", "Var"])

    for var in variables:
        data = pd.to_numeric(df[var], errors='coerce')
        mode_value = data.mode().iloc[0] if not data.mode().empty else None
        result = {
            "Variable": var,
            "Unique_Count": data.nunique(),
            "Min": data.min(),
            "Max": data.max(),
            "Mean": data.mean(),
            "Median": data.median(),
            "Mode": mode_value,
            "SD": data.std(),
            "Var": data.var()
        }
        results_table = results_table.append(result, ignore_index=True)

    # Ensure all numeric columns are of the same type and round to 2 decimal places
    numeric_columns = ["Unique_Count", "Min", "Max", "Mean", "Median", "Mode", "SD", "Var"]
    results_table[numeric_columns] = results_table[numeric_columns].astype(float).round(2)

    return results_table

# COMMAND ----------

# List the variables of interest for evaporator stage
evaporator_vars = [
    "EVAPORATOR_FEED_FLOW",
    "EVAPORATOR_EFFECT_1_HEATING_TEMPERATURE",
    "EVAPORATOR_EFFECT_1_BOILING_TEMPERATURE",
    "EVAPORATOR_EFFECT_1_TEMPERATURE_DIFFERENTIAL",
    "TVR_STEAM_PRESSURE",
    "EVAPORATOR_CONCENTRATE_CALC_TS",
    "EVAPORATOR_CONCENTRATE_FLOWRATE",
    "EVAPORATOR_CONCENTRATE_DENSITY",
    "EVAPORATOR_COOLING_WATER_TEMP",
    "FLASH_VESSEL_NUMBER",
    "SECOND_STAGE_DSI_TEMPERATURE"
]
# Calculate statistics for the evaporator variables
results_table = calculate_statistics(filtered_df, evaporator_vars)

# Display the table
display(results_table)

# COMMAND ----------

# MAGIC %md
# MAGIC FLASH_VESSEL_NUMBER is an categorical variable.

# COMMAND ----------

# Plot the distribution of FLASH_VESSEL_NUMBER VS TEST_ACTUAL_RESULT
plot_categorical_summary(filtered_df, 'FLASH_VESSEL_NUMBER', 'TEST_ACTUAL_RESULT', legend_loc='upper left')

# COMMAND ----------

def create_faceted_boxplot(df, id_var, value_vars, hue_var):
    """
    Create a faceted boxplot for a given set of variables.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    id_var (str): The identifier variable.
    value_vars (list): A list of variable names to plot.
    hue_var (str): The variable to use for hue.
    
    Returns:
    sns.FacetGrid: The FacetGrid object.
    pd.DataFrame: The DataFrame containing the calculated statistics.
    """
    # Select columns of interest and pivot longer
    melted_df = df.melt(
        id_vars=[id_var],
        value_vars=value_vars,
        var_name='Variable',
        value_name='Value'
    )

    # Ensure all values in the 'Value' column are numeric
    melted_df['Value'] = pd.to_numeric(melted_df['Value'], errors='coerce')

    # Drop rows with NaN values in the 'Value' column
    melted_df = melted_df.dropna(subset=['Value'])

    # Calculate statistics
    stats = melted_df.groupby([id_var, 'Variable']).agg(
        Min_Value=('Value', 'min'),
        Max_Value=('Value', 'max'),
        Mean_Value=('Value', 'mean'),
        Median_Value=('Value', 'median')
    ).reset_index().round(2)

    # Create a faceted boxplot
    g = sns.FacetGrid(melted_df, col='Variable', col_wrap=3, sharey=False, height=4)
    g.map_dataframe(sns.boxplot, x=id_var, y='Value', hue=hue_var, palette='husl')

    # Add annotations for the median value
    for ax in g.axes.flat:
        lines = ax.get_lines()
        categories = ax.get_xticks()
        
        for cat in categories:
            median_line = lines[4 + cat * 6]  # Median line
            median_value = round(median_line.get_ydata()[0], 2)  # Round to two decimal places
            
            ax.text(
                cat, 
                median_value, 
                f'{median_value}', 
                ha='center', 
                va='center', 
                fontweight='bold', 
                size=7,
                color='white',
                bbox=dict(facecolor='#445A64')
            )

    return g, stats


# COMMAND ----------

# Create a faceted boxplot show the relation between TEST_ACTUAL_RESULT and the evaporator numerical variables
g, stats = create_faceted_boxplot(
    filtered_df, 
    'TEST_ACTUAL_RESULT', 
    [var for var in evaporator_vars if var != 'FLASH_VESSEL_NUMBER'], 
    'TEST_ACTUAL_RESULT'
)

# Display the statistics table about the evaporator variables VS TEST_ACTUAL_RESULT
display(stats)

g.add_legend()
g.set_titles("{col_name}")
g.set_axis_labels("TEST_ACTUAL_RESULT", "Value")
g.fig.suptitle("Variables vs. TEST_ACTUAL_RESULT Value", fontsize=14, fontweight='bold')
g.fig.subplots_adjust(top=0.9)
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ##  3. OIL ADDITION & CONCENTRATE PREPARATION

# COMMAND ----------

# MAGIC %md
# MAGIC For this step, we do descriptive statistic analysis for the oil addition and concentrate preparation stage processing parameters. includes:
# MAGIC - OIL_DIVERT_TEMPERATURE
# MAGIC - OIL_DOSING_FLOWRATE
# MAGIC - OIL_DOSING_CORRECTION_FACTOR
# MAGIC - DRYER_FEEDLINE_PRESSURE
# MAGIC - DRYER_FEEDLINE_TEMPERATURE
# MAGIC - HPP_MOTOR_SPEED
# MAGIC - HOMOGENISER_STAGE_1_PRESSURE
# MAGIC - HOMOGENISER_STAGE_2_PRESSURE
# MAGIC - DRYER_CONCENTRATE_FEED_FLOWRATE
# MAGIC - DRYER_CONCENTRATE_FEED_FLOWRATE_POST_OIL
# MAGIC - DRYER_CONCENTRATE_CALC_TS_POST_OIL
# MAGIC
# MAGIC Then analyze the relationship between each processing parameter and the target variable which is the Flecks actual test result.
# MAGIC

# COMMAND ----------

# List the variables of interest for oil addition & concentrate preparation
oil_vars = [
    "OIL_DIVERT_TEMPERATURE",
    "OIL_DOSING_FLOWRATE",
    "OIL_DOSING_CORRECTION_FACTOR",
    "DRYER_FEEDLINE_PRESSURE",
    "DRYER_FEEDLINE_TEMPERATURE",
    "HPP_MOTOR_SPEED", 
    "HOMOGENISER_STAGE_1_PRESSURE",
    "HOMOGENISER_STAGE_2_PRESSURE",
    "DRYER_CONCENTRATE_FEED_FLOWRATE",
    "DRYER_CONCENTRATE_FEED_FLOWRATE_POST_OIL",
    "DRYER_CONCENTRATE_CALC_TS_POST_OIL"   

]

# Calculate statistics for the variables
results_table = calculate_statistics(filtered_df, oil_vars)

# Display the table
display(results_table)

# COMMAND ----------

# Create a faceted boxplot show the relation between TEST_ACTUAL_RESULT and the oil addition & concentrate preparation numerical variables
g, stats = create_faceted_boxplot(filtered_df, 'TEST_ACTUAL_RESULT', oil_vars, 'TEST_ACTUAL_RESULT')

# Display the statistics table for the oil addition & concentrate preparation variables VS TEST_ACTUAL_RESULT
display(stats)

g.add_legend()
g.set_titles("{col_name}")
g.set_axis_labels("TEST_ACTUAL_RESULT", "Value")
g.fig.suptitle("Variables vs. TEST_ACTUAL_RESULT Value", fontsize=14, fontweight='bold')
g.fig.subplots_adjust(top=0.9)
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. SPRAY DRYING STAGE (DRYER)    

# COMMAND ----------

# MAGIC %md
# MAGIC For this step, we do descriptive statistic analysis for the spray drying stage processing parameters. includes:
# MAGIC - DRYER_INLET_HUMIDITY
# MAGIC - DRYER_INLET_AIR_TEMPERATURE
# MAGIC - DRYER_OUTLET_AIR_TEMPERATURE
# MAGIC - DRYER_INLET_FAN_SPEED
# MAGIC - DRYER_INLET_AIR_FLOWRATE
# MAGIC - EFB_POWDER_TEMPERATURE
# MAGIC - EXHAUST_ABSOLUTE_HUMIDITY
# MAGIC - EXHAUST_RELATIVE_HUMIDITY
# MAGIC - DRYER_FEEDLINE_NUMBER
# MAGIC - CONCENTRATE_TANK_NUMBER
# MAGIC - POWDER_OUTPUT_RATE
# MAGIC - CONCENTRATE_FILTER_CHANGE
# MAGIC - DRYER_TIME_ON_PRODUCT_SINCE_CIP
# MAGIC - FINES_LINE_A_PRESSURE
# MAGIC - FINES_LINE_B_PRESSURE
# MAGIC - CYCLONE_1_TEMP_DIFFERENTIAL
# MAGIC - CYCLONE_2_TEMP_DIFFERENTIAL
# MAGIC - CYCLONE_1_SPLITTERS_CV_POSITION
# MAGIC - CYCLONE_2_SPLITTERS_CV_POSITION
# MAGIC - IFB_AIR_PRESSURE
# MAGIC - IFB_AIR_TEMPERATURE
# MAGIC
# MAGIC Then analyze the relationship between each processing parameter and the target variable which is the Flecks actual test result.

# COMMAND ----------

# List the variables of interest for drying stage part 1
drying_vars = [
  "DRYER_INLET_HUMIDITY",
  "DRYER_INLET_AIR_TEMPERATURE",
  "DRYER_OUTLET_AIR_TEMPERATURE",
  "DRYER_INLET_FAN_SPEED",
  "DRYER_INLET_AIR_FLOWRATE",
  "EFB_POWDER_TEMPERATURE",
  "EXHAUST_ABSOLUTE_HUMIDITY",
  "EXHAUST_RELATIVE_HUMIDITY",
  "DRYER_FEEDLINE_NUMBER",
  "CONCENTRATE_TANK_NUMBER",
  "POWDER_OUTPUT_RATE",
  "CONCENTRATE_FILTER_CHANGE"
]

# Calculate statistics for the variables
results_table = calculate_statistics(filtered_df, drying_vars)

# Display the table
display(results_table)

# COMMAND ----------

# MAGIC %md
# MAGIC DRYER_FEEDLINE_NUMBER, CONCENTRATE_TANK_NUMBER, and CONCENTRATE_FILTER_CHANGE are categorical variables.

# COMMAND ----------

# DRYER_FEEDLINE_NUMBER VS TEST_ACTUAL_RESULT
plot_categorical_summary(filtered_df, 'DRYER_FEEDLINE_NUMBER', 'TEST_ACTUAL_RESULT', legend_loc='upper left')

# COMMAND ----------

# CONCENTRATE_TANK_NUMBER VS TEST_ACTUAL_RESULT
plot_categorical_summary(filtered_df, 'CONCENTRATE_TANK_NUMBER', 'TEST_ACTUAL_RESULT', legend_loc='upper left')

# COMMAND ----------

# CONCENTRATE_FILTER_CHANGE VS TEST_ACTUAL_RESULT
plot_categorical_summary(filtered_df, 'CONCENTRATE_FILTER_CHANGE', 'TEST_ACTUAL_RESULT', legend_loc='upper left')

# COMMAND ----------

# List the variables of interest after filtering DRYER_FEEDLINE_NUMBER, CONCENTRATE_TANK_NUMBER, and CONCENTRATE_FILTER_CHANGE
drying_vars_filtered = [
  "DRYER_INLET_HUMIDITY",
  "DRYER_INLET_AIR_TEMPERATURE",
  "DRYER_OUTLET_AIR_TEMPERATURE",
  "DRYER_INLET_FAN_SPEED",
  "DRYER_INLET_AIR_FLOWRATE",
  "EFB_POWDER_TEMPERATURE",
  "EXHAUST_ABSOLUTE_HUMIDITY",
  "EXHAUST_RELATIVE_HUMIDITY",
  "POWDER_OUTPUT_RATE"
]

# Create a faceted boxplot show the relation between TEST_ACTUAL_RESULT and the drying stage part 1 numerical variables
g, stats = create_faceted_boxplot(filtered_df, 'TEST_ACTUAL_RESULT', drying_vars_filtered, 'TEST_ACTUAL_RESULT')

# Display the statistics table for the drying stage part 1 variables VS TEST_ACTUAL_RESULT
display(stats)

g.add_legend()
g.set_titles("{col_name}")
g.set_axis_labels("TEST_ACTUAL_RESULT", "Value")
g.fig.suptitle("Variables vs. TEST_ACTUAL_RESULT Value", fontsize=14, fontweight='bold')
g.fig.subplots_adjust(top=0.9)
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

# List the variables of interest for stage drying part 2
drying_part2_vars = [
  "DRYER_TIME_ON_PRODUCT_SINCE_CIP",
  "FINES_LINE_A_PRESSURE",
  "FINES_LINE_B_PRESSURE",
  "CYCLONE_1_TEMP_DIFFERENTIAL",
  "CYCLONE_2_TEMP_DIFFERENTIAL",
  "CYCLONE_1_SPLITTERS_CV_POSITION",
  "CYCLONE_2_SPLITTERS_CV_POSITION",
  "IFB_AIR_PRESSURE",
  "IFB_AIR_TEMPERATURE"
]

# Calculate statistics for the variables
results_table = calculate_statistics(filtered_df, drying_part2_vars)

# Display the table
display(results_table)

# COMMAND ----------

# Create a faceted boxplot show the relation between TEST_ACTUAL_RESULT and the stage drying part 2 numerical variables
g, stats = create_faceted_boxplot(filtered_df, 'TEST_ACTUAL_RESULT', drying_part2_vars, 'TEST_ACTUAL_RESULT')

# Display the statistics table for the stage drying part 2 variables VS TEST_ACTUAL_RESULT
display(stats)

g.add_legend()
g.set_titles("{col_name}")
g.set_axis_labels("TEST_ACTUAL_RESULT", "Value")
g.fig.suptitle("Variables vs. TEST_ACTUAL_RESULT Value", fontsize=14, fontweight='bold')
g.fig.subplots_adjust(top=0.9)
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. EXTERNAL FLUID BED & NOZZLE SETTINGS     

# COMMAND ----------

# MAGIC %md
# MAGIC For this step, we do descriptive statistic analysis for the external fluid bed and nozzle settings stage processing parameters. includes:
# MAGIC -   EFB_SECTION_1_AIR_PRESSURE
# MAGIC -   EFB_SECTION_2_AIR_PRESSURE
# MAGIC -   EFB_SECTION_3_AIR_PRESSURE
# MAGIC -   EFB_1_AIR_TEMPERATURE_TARGET
# MAGIC -   EFB_2_AIR_TEMPERATURE_TARGET
# MAGIC -   EFB_3_AIR_TEMPERATURE_TARGET
# MAGIC -   POWDER_TRANSPORT_BREAKDOWN
# MAGIC -   LANCE_PITCH_INNER_LANCES
# MAGIC -   LANCE_PITCH_OUTER_LANCES
# MAGIC -   NOZZLE_ORIFICE_INNER_LANCES
# MAGIC -   NOZZLE_ORIFICE_OUTER_LANCES
# MAGIC -   NOZZLE_SWIRL_INNER_LANCES
# MAGIC -   NOZZLE_SWIRL_OUTER_LANCES
# MAGIC -   BLOWER_A_SPEED
# MAGIC -   BLOWER_B_SPEED
# MAGIC -   FINES_A_DESTINATION
# MAGIC -   FINES_B_DESTINATION
# MAGIC
# MAGIC Then analyze the relationship between each processing parameter and the target variable which is the Flecks actual test result.

# COMMAND ----------

# List the variables of interest for external part
external_vars = [
  "EFB_SECTION_1_AIR_PRESSURE",
  "EFB_SECTION_2_AIR_PRESSURE",
  "EFB_SECTION_3_AIR_PRESSURE",
  "EFB_1_AIR_TEMPERATURE_TARGET",
  "EFB_2_AIR_TEMPERATURE_TARGET",
  "EFB_3_AIR_TEMPERATURE_TARGET",
  "POWDER_TRANSPORT_BREAKDOWN", 
  "LANCE_PITCH_INNER_LANCES",
  "LANCE_PITCH_OUTER_LANCES",
  "NOZZLE_ORIFICE_INNER_LANCES",
  "NOZZLE_ORIFICE_OUTER_LANCES",
  "NOZZLE_SWIRL_INNER_LANCES",
  "NOZZLE_SWIRL_OUTER_LANCES",
  "BLOWER_A_SPEED",
  "BLOWER_B_SPEED",
  "FINES_A_DESTINATION",
  "FINES_B_DESTINATION"
]

# Calculate the number of unique values for each variable
unique_table = show_unique_values(filtered_df, external_vars)

# Display the table
display(unique_table)


# COMMAND ----------

# MAGIC %md
# MAGIC For this parts, all variables are categorical variables

# COMMAND ----------

# Plot each variable against TEST_ACTUAL_RESULT
for var in external_vars:
    plot_categorical_summary(filtered_df, var, 'TEST_ACTUAL_RESULT')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. FINAL POWDER PROPERTIES & APPEARANCE

# COMMAND ----------

# MAGIC %md
# MAGIC For this step, we do descriptive statistic analysis for the external fluid bed and nozzle settings stage processing parameters. includes:
# MAGIC - PROCESS_BULK_DENSITY
# MAGIC - PACKING_BULK_DENSITY
# MAGIC - POWDER_APPEARANCE_TEST_METHOD
# MAGIC - REWORK_ADDED_QUANTITY
# MAGIC
# MAGIC Then analyze the relationship between each processing parameter and the target variable which is the Flecks actual test result.

# COMMAND ----------

# List the variables of interest for the density variables
density_vars = [
  "PROCESS_BULK_DENSITY",
  "PACKING_BULK_DENSITY"
]

# Calculate statistics for the variables
results_table = calculate_statistics(filtered_df, density_vars)

# Display the table
display(results_table)

# COMMAND ----------

# Create a faceted boxplot show the relation between TEST_ACTUAL_RESULT and the density numerical variables
g, stats = create_faceted_boxplot(filtered_df, 'TEST_ACTUAL_RESULT', density_vars, 'TEST_ACTUAL_RESULT')

# Display the statistics table for the density variables VS TEST_ACTUAL_RESULT
display(stats)

g.add_legend()
g.set_titles("{col_name}")
g.set_axis_labels("TEST_ACTUAL_RESULT", "Value")
#g.fig.suptitle("Variables vs. TEST_ACTUAL_RESULT Value", fontsize=10, fontweight='bold')
g.fig.subplots_adjust(top=0.9)
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

# List the variables of interest for appearance variables
appearance_vars = [
  "POWDER_APPEARANCE_TEST_METHOD",
  "REWORK_ADDED_QUANTITY"
]

# Calculate statistics for the variables
unique_table = show_unique_values(filtered_df, appearance_vars)

# Display the table
display(unique_table)

# COMMAND ----------

# Plot each variable against TEST_ACTUAL_RESULT
for var in appearance_vars:
    plot_categorical_summary(filtered_df, var, 'TEST_ACTUAL_RESULT')