Define the methodology to be used for extracting insights

Step 1: Data Preparation
Extract all rows corresponding to SKU_NUMBER = "YCF325" And lab result PPP variables. 

'Scorched Particles (DV)' 'Foam Height (DV)' 'Scorched Particles (BS 1743)' 'Bulk Density' 'Tumbler (DV)' 'Reconstitution (DV)' 'Free Fat NIR In-house' 'Peroxide Value GCPVAL04 01' 'Insolubility Index (modified)' 'Flecks' 'Wettability at 40°C (modified)' 'Bulk Volume (DV)'

Data Cleaning & Formatting: 
Convert columns from objects to appropriate data formats.
Handle missing values by either imputation or exclusion, depending on the nature and frequency of missingness.
Put the Optimal min, Optimal max, and Target value for each PPP in the table
Step 2: Exploratory Data Analysis (EDA)
Descriptive Statistics:
Compute summary statistics (mean, median, standard deviation) for PPP metrics and process parameters.
Distribution & Variability:
Plot histograms or boxplots to understand the distribution of PPP metrics and identify outliers or skewed distributions.
Correlation Analysis:
Use correlation matrices (Pearson/Spearman) to identify which process parameters (e.g., dryer inlet temperature, evaporator concentrate density, homogenizer pressures, nozzle settings) correlate most strongly with key PPP metrics like bulk density. From paper 
Step 3: Feature Selection & Data Transformation
Key Variable Identification:
Select a subset of high-impact parameters from the correlation analysis as candidate predictors.
Feature Engineering:
Consider creating derived features (e.g., temperature differentials, air flow ratios to feed flow) if these provide additional predictive power.
Dimensionality Reduction (Optional):
If the parameter set is large and complex, apply Principal Component Analysis technique to reduce dimensionality, while ensuring the interpretability of results remains possible.
Step 4: Modeling to Identify Optimal Ranges
Model Selection:
Begin with interpretable models such as multiple linear regression to understand linear relationships.
For more complex relationships, consider nonlinear models (Random Forest, Gradient Boosted Trees, or Neural Networks) that can capture interactions between parameters.
Training & Validation:
Split data into training and test sets (e.g., 70/30 split) or use cross-validation to ensure the model generalizes well.
Model Interpretation:
For tree-based models, use feature importance or SHAP values to understand which parameters most influence the target PPP metrics.
Analyze partial dependence plots or sensitivity analyses to understand how changing each parameter affects the predicted outcome.
Step 5: Optimization of Process Parameters
Scenario Testing:
Once the predictive model is established, define desired target ranges for PPP outcomes (e.g., target bulk density, acceptable moisture level).
Parametric Optimization:
Use optimization techniques to vary process parameters within feasible ranges to find the combination that produces the desired PPP targets.
This can be as simple as a grid search over parameter ranges or as sophisticated as employing optimization algorithms (e.g., gradient-based optimizers or genetic algorithms).
Trade-Off Analysis:
If multiple PPP outcomes are important simultaneously (e.g., achieve a certain bulk density while minimizing scorched particles), perform multi-objective optimization or scenario simulations to find a balance.
Step 6: Validation & Recommendations
Historical Validation:
Check historical runs where process parameters fell into the model-recommended ranges and confirm that those runs achieved the desired PPP outcomes.
Pilot Adjustments:
Propose adjustments to dryer inlet temperatures, homogeniser pressures, or nozzle settings based on the model’s insights.
Validate these adjustments through pilot runs or controlled experiments.
Continuous Improvement:
As new data becomes available (e.g., new production runs), retrain and update the model to refine the optimal ranges.
Deliverables
Descriptive Reports: Summaries of current PPP performance for SKU YCF325 and correlations to process parameters.
Predictive Model: A validated model (e.g., Random Forest) hosted in a data platform like Snowflake or on a local machine.
Optimization Guidelines: Parameter setpoints and ranges are recommended to consistently achieve optimal PPP metrics.
