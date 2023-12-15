# Databricks notebook source
spark

# COMMAND ----------

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


# COMMAND ----------

import os
from pyspark.sql.functions import col, isnull, when, count, udf

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

filename="landing/fhvhv_tripdata_2019-05.parquet"
file_path = 's3a://' + bucket_name +"/" + filename
sdf = spark.read.parquet(file_path)

# COMMAND ----------

sdf.head(10)

# COMMAND ----------

sdf.printSchema()

# COMMAND ----------

sdf.show()

# COMMAND ----------

sdf.count()

# COMMAND ----------

sdf.dropDuplicates()
sdf.count()

# COMMAND ----------

#samples=sdf.sample(False,0.25)
#samples.summary().show()

# COMMAND ----------

#samples.select([count(when(isnull(c), c)).alias(c) for c in samples.columns]).show()

# COMMAND ----------

#drop dispatch_base_num and originating_base_num
sdf=sdf.drop("dispatching_base_num","originating_base_num")
sdf.show(5)

# COMMAND ----------

sdf.select('airport_fee').show()

# COMMAND ----------

#changing null values to 0 in airport fee
sdf = sdf.withColumn('airport_fee', when(sdf['airport_fee'].isNull(), 0).otherwise(sdf['airport_fee']))
sdf.select('airport_fee').show(10)

# COMMAND ----------

#check summary
sdf.summary()

# COMMAND ----------

sdf.select([count(when(isnull(c), c)).alias(c) for c in sdf.columns]).show()

# COMMAND ----------

sdf.select('wav_match_flag').show()

# COMMAND ----------

sdf=sdf.drop('shared_request_flag','shared_match_flag','access_a_ride_flag','wav_request_flag','wav_match_flag')

# COMMAND ----------

sdf.printSchema()

# COMMAND ----------

#transform time data to year,month,date, day of month, day of week, hour, minute, second

sdf=sdf.withColumn("pickup_year",F.year('pickup_datetime'))
sdf=sdf.withColumn("pickup_month",F.month('pickup_datetime'))
sdf=sdf.withColumn("pickup_day_of_month",F.dayofmonth('pickup_datetime'))
sdf=sdf.withColumn("pickup_day_of_week",F.dayofweek('pickup_datetime'))
sdf=sdf.withColumn("pickup_hour",F.hour('pickup_datetime'))
sdf=sdf.withColumn("pickup_minute",F.minute('pickup_datetime'))
sdf=sdf.withColumn("pickup_second",F.second('pickup_datetime'))
sdf.printSchema()

# COMMAND ----------

#transform time data to year,month,date, day of month, day of week, hour, minute, second
sdf=sdf.withColumn("dropoff_year",F.year('dropoff_datetime'))
sdf=sdf.withColumn("dropoff_month",F.month('dropoff_datetime'))
sdf=sdf.withColumn("dropoff_day_of_month",F.dayofmonth('dropoff_datetime'))
sdf=sdf.withColumn("dropoff_day_of_week",F.dayofweek('dropoff_datetime'))
sdf=sdf.withColumn("dropoff_hour",F.hour('dropoff_datetime'))
sdf=sdf.withColumn("dropoff_minute",F.minute('dropoff_datetime'))
sdf=sdf.withColumn("dropoff_second",F.second('dropoff_datetime'))
sdf.printSchema()

# COMMAND ----------

sdf.show(10)

# COMMAND ----------

sdf=sdf.drop('request_datetime','on_scene_datetime','dropoff_datetime','pickup_datetime')

# COMMAND ----------

#add pickup_weekend
sdf=sdf.withColumn("pickup_weekend", when(sdf.pickup_day_of_week==6,1.0).when(sdf.pickup_day_of_week==7,1.0).otherwise(0))

#add dropoff_weekend
sdf=sdf.withColumn("dropoff_weekend", when(sdf.dropoff_day_of_week==6,1.0).when(sdf.dropoff_day_of_week==7,1.0).otherwise(0))


# COMMAND ----------

sdf.show(5)

# COMMAND ----------

index=filename.index('/')+1
filename = 'cleaned_'+filename[index:]
filename

# COMMAND ----------

output_file_path="s3://pp-nyc-trips-dats/raw/"+filename
output_file_path

# COMMAND ----------

sdf = sdf.repartition(1)
sdf.write.parquet(output_file_path)

# COMMAND ----------

len(sdf.columns)
