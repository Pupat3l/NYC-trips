# Databricks notebook source
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import sklearn
import numpy
import scipy
import plotly
import bs4 as bs
import urllib.request
import boto3
import pyspark.sql.functions as F
import os
from pyspark.sql.functions import col, isnull, when, count, udf
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
# Import the evaluation module
from pyspark.ml.evaluation import *
# Import the model tuning module
from pyspark.ml.tuning import *


# COMMAND ----------

spark

# COMMAND ----------

# To work with Amazon S3 storage, set the following variables using your AWS Access Key and Secret Key
# Set the Region to where your files are stored in S3.
access_key = 'xyz'
secret_key = 'xyz'

# COMMAND ----------

# Set the environment variables so boto3 can pick them up later
os.environ['AWS_ACCESS_KEY_ID'] = access_key
os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
encoded_secret_key = secret_key.replace("/", "%2F")
aws_region = "us-east-2"
bucket_name = "pp-nyc-trips-data"

# COMMAND ----------

# Update the Spark options to work with our AWS Credentials
sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", access_key)
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", secret_key)
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3." + aws_region +
".amazonaws.com")
sc._jsc.hadoopConfiguration().set("fs.s3a.bucket."+bucket_name +".endpoint.region", aws_region)

# COMMAND ----------

filename="raw/cleaned_fhvhv_tripdata_2021-05.parquet"
file_path = 's3a://' + bucket_name +"/" + filename
sdf = spark.read.parquet(file_path)

# COMMAND ----------

sdf.count()

# COMMAND ----------

len(sdf.columns)

# COMMAND ----------

training,test=sdf.randomSplit([0.70,0.30], seed=45)

# COMMAND ----------

indexer = StringIndexer(inputCol='hvfhs_license_num',outputCol='licenseIndex', handleInvalid="keep")
indexed_sdf=indexer.fit(sdf).transform(sdf)

# COMMAND ----------

integer_cols = [col_name for col_name, col_type in indexed_sdf.dtypes if col_type == 'int']
integer_cols

# COMMAND ----------

double_cols = [col_name for col_name,col_type in indexed_sdf.dtypes if (col_type == 'double' and col_name != 'tips')]
double_cols

# COMMAND ----------

out_cols=[name+'Vector' for name in integer_cols]
out_cols

# COMMAND ----------

#encode all columns
encoder = OneHotEncoder(inputCols=integer_cols,outputCols=out_cols, dropLast=True,handleInvalid="keep")
encoded_sdf = encoder.fit(indexed_sdf).transform(indexed_sdf)
encoded_sdf

# COMMAND ----------

input_cols=out_cols+double_cols
input_cols

# COMMAND ----------

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")

# COMMAND ----------

# Create a Linear Regression Estimator
linear_reg = LinearRegression(labelCol='tips')
# Create a regression evaluator (to get RMSE, R2, RME, etc.)
evaluator = RegressionEvaluator(labelCol='tips')


# COMMAND ----------

# Create the pipeline   Indexer is stage 0 and Linear Regression (linear_reg)  is stage 3
regression_pipe = Pipeline(stages=[indexer, encoder, assembler, linear_reg])

# COMMAND ----------

# Create a grid to hold hyperparameters 
grid = ParamGridBuilder()

# Build the parameter grid
grid = grid.build()

# COMMAND ----------


# Create the CrossValidator using the hyperparameter grid
cv = CrossValidator(estimator=regression_pipe, 
                    estimatorParamMaps=grid, 
                    evaluator=evaluator, 
                    numFolds=3)


# COMMAND ----------

# Train the models
all_models  = cv.fit(training)

# COMMAND ----------

# Show the average performance over the three folds
print(f"Average metric {all_models.avgMetrics}")


# COMMAND ----------

# Get the best model from all of the models trained
bestModel = all_models.bestModel

# COMMAND ----------

# Use the model 'bestModel' to predict the test set
test_results = bestModel.transform(test)

# COMMAND ----------

# Show the predicted tip
test_results.select('tips', 'prediction').show(truncate=False)

# COMMAND ----------

# Calculate RMSE and R2
rmse = evaluator.evaluate(test_results, {evaluator.metricName:'rmse'})
r2 =evaluator.evaluate(test_results,{evaluator.metricName:'r2'})
print(f"RMSE: {rmse}  R-squared:{r2}")

# COMMAND ----------

model_name="tip_prediction_model_"+filename[-13:-8]
model_path="s3://pp-nyc-trips-data/models/"+model_name
model_path

# COMMAND ----------

# Save the model to S3
bestModel.save(model_path)

# COMMAND ----------

test_results

# COMMAND ----------

sdf = test_results.repartition(1)
sdf

# COMMAND ----------

output_file_path='s3://pp-nyc-trips-data/trusted/test_results_'+filename[-13:-8]
output_file_path

# COMMAND ----------

sdf.write.parquet(output_file_path)

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

# The Spark dataframe test_results holds the original 'tip' as well as the 'prediction'
# Select and convert to a Pandas dataframe
df = test_results.select('tips','prediction','base_passenger_fare','dropoff_day_of_week').toPandas()
df

# COMMAND ----------

day_mapping = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
df['dropoff_day_of_week'] = df['dropoff_day_of_week'].map(day_mapping)
df

# COMMAND ----------

# Make sure the days column is a categorical variable for proper ordering on the x-axis
df['dropoff_day_of_week'] = pd.Categorical(df['dropoff_day_of_week'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
df

# COMMAND ----------

df_filtered = df[(df['base_passenger_fare'] <= 40) & (df['tips'] <= 100)]
df_filtered

# COMMAND ----------

grouped_df = df_filtered.groupby('dropoff_day_of_week')[['tips', 'prediction']].mean()

# COMMAND ----------

grouped_df

# COMMAND ----------

ax = grouped_df.plot(kind='bar', figsize=(10, 6))
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Average Tips')
ax.set_title('Comparison of Tips and Predicted Tips for Each Day of the Week')
plt.show()


# COMMAND ----------

# Plot a line chart
ax = grouped_df[['tips', 'prediction']].plot(kind='line', marker='o', figsize=(10, 6))

# Add labels and title
plt.xlabel('Day of the Week')
plt.ylabel('Tips')
plt.title('Comparison of Tips and Predicted Tips for Each Day of the Week')

# Show the legend
plt.legend(["Actual Tips", "Predicted Tips"])

# Show the plot
plt.show()

# COMMAND ----------

sample, big = test_results.randomSplit([0.1,0.9], seed=42)

# COMMAND ----------

residual_df=sample.select('tips','prediction','base_passenger_fare').toPandas()
residual_df

# COMMAND ----------

residual_df=residual_df[(residual_df['base_passenger_fare'] <= 40) & (residual_df['tips'] <= 100)]
residual_df

# COMMAND ----------

# Calculate residuals
residual_df['residuals'] = residual_df['tips'] - residual_df['prediction']

# Create a residual plot
plt.figure(figsize=(10, 6))
sns.histplot(residual_df['residuals'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Residuals: Actual Tips vs Predicted Tips')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

fares_cols=['base_passenger_fare','tolls','bcf','sales_tax','congestion_surcharge','airport_fee','driver_pay','tips','prediction','dropoff_day_of_week']

# COMMAND ----------

sample, big = test_results.randomSplit([0.1,0.9], seed=42)

# COMMAND ----------

df=sample.select(fares_cols).toPandas()

# COMMAND ----------

check=df[(df['base_passenger_fare'] <= 40) & (df['tips'] <= 100)]

# COMMAND ----------

check['total']=check[['base_passenger_fare','tolls','bcf','sales_tax','congestion_surcharge','airport_fee']].sum(axis=1)
check 


# COMMAND ----------

# Map integer values to day names
check['day_name'] = check['dropoff_day_of_week'].map({1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'})

# COMMAND ----------

check = check.groupby('day_name').mean()


# COMMAND ----------

check['tip_ratio']=check['tips']/check['total']
check['prediction_ratio']=check['prediction']/check['total']

# COMMAND ----------

check

# COMMAND ----------

grouped_df = check.groupby('day_name').mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
grouped_df

# COMMAND ----------

# Plot grouped bar chart for selected columns
ax = grouped_df[['tip_ratio','prediction_ratio']].plot(kind='bar', figsize=(10, 6), width=0.2)

# Add labels and title
plt.xlabel('Day of Week')
plt.ylabel('Values')
plt.title('Comparison of Selected Columns for Each Category')

# Show the legend
plt.legend(['tip_ratio','prediction_ratio'])

# Show the plot
plt.show()

# COMMAND ----------


# Assuming df is your DataFrame with columns 'tip' and 'prediction'
sns.lmplot(x='tips', y='prediction', data=check)
plt.xlabel('Actual Tips')
plt.ylabel('Predicted Tips')
plt.title('Scatter Plot: Actual Tips vs. Predicted Tips')
plt.show()

