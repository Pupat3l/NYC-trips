# NYC-trips
## A big Data Project by Pujan Patel

## Description
This project is a creation of machine learning pipeline that incorporates big data technologies using a cloud infrastructure. The project consolidates Extraction, EDA, Transformation, Feature Engineering and Modeling, and Data Visualization to create a machine learning model that uses For-Hire Vehicle data from [Kaggle](https://www.kaggle.com/) to create a tip predictor for rides. The For-Hire Vehicle industries includes all taxis that offer passenger pick ups and drop off services. It includes yellow cab, Uber, Lyft, Revel etc. A rider's fare includes the fare based on distance and time, taxes, tolls, fees, and other charges if applicable. The driver receives fares as payment and tip on top based on customer's choice. The goal of this project as mentioned before is to create a tip predictor based of the data like trip time, trip distance, fare, tolls, fees etc.

## Data 
The dataset that I have decided to utilize for the final project is called "NYC FHV(Uber/Lyft) Trip Data Expanded (2019-2022)" and can be found [here](https://www.kaggle.com/datasets/jeffsinsel/nyc-fhvhv-data). The dataset provides detailed information about every For-Hire Vehicle (which includes Uber and Lyft) rides in the New York City. The dataset was originally found on the official nyc.gov website. Information regarding the trip included in the dataset is TLC license number, date and time of the pick-up and drop-off, taxi zone, passenger pick-up and drop-off time, and tip etc. The dataset also has information regarding the distance travelled, NYC taxes, tolls, and surcharges for the LGA and JFK airport. Overall, it includes all the data generated from a ride.

Talking about the dataset, the dataset has potential for many analyses and predictions. However, I was considering one of two options which were tip forecasting or revenue forecasting. Tip forecasting would aim to predict the amount of tip a customer will give based on the variables like trip time, trip miles, trip cost, tax, airport fees, tolls, etc. Revenue forecasting is slightly easier where it would focus on generating predictions for revenues based on historic data i.e. revenue trend on the previous rides multiplied by an inflation index to get a proximate ride today. Sticking to only one of these two option, my main priority would be the tip forecasting.

## Methods
### Data Acquistion (Extraction): 
- Extracting the data from the source and storing at a temporary cloud storage. 
- I used AWS S3 bucket service to store the dataset and organized the bucket in folders like landing, raw, trusted, and models. 
- Used -Curl in AWS CLI to perform that

```
#downloaded file
kaggle datasets download -d jeffsinsel/nyc-fhvhv-data -f
fhvhv_tripdata_2019-02.parquet
#unzip 
unzip fhvhv_tripdata_2019-02.parquet
#copying to s3
aws s3 cp fhvhv_tripdata_2019-02.parquet s3://pp-nyc-trips-
data/landing/fhvhv_tripdata_2019-02.parquet
#checking if its visible in s3
aws s3 ls s3://pp-nyc-trips-data/landing/
#yes 
#removed the files from the local system
rm fhvhv_tripdata_2019-02.parquet.zip
rm fhvhv_tripdata_2019-02.parquet

```

### Exploratory Data Analysis: 
- Read the dataset from landing and learn about the data so then transformation can be performed easily.

### Feature Engineering and Modeling: 
- Utilized Databricks Community version for using Hadoop Cluster and jupyter notebook.

#### Data Cleaning:
- Cleaned data using pyspark libary in python from landing folder.
- Transported the data to S3 bucket in raw folder storing cleaned dataset to use it further.
- The script used was [Clean.py](https://github.com/Pupat3l/NYC-trips/blob/main/clean_pyspark.py)

#### Feature Engineering:
- Cleaned data, from raw folder of the s3 bucket, was used followed by using relevant columns (trip duration, fare, tax, fee etc) only.
- Used StringIndexer, OneHotEncodder, VectorAssembler to create features that includes vectors from all columns but tips.

#### Modelimn:
- Created a LinearRegression on 'tips' which is the target feature, evaluator using RegressionEvaluator.
- Then created a pipeline using indexer, encoder, assembler, and regression. 
- Then the dataset was already splitted into two parts: training data(0.70), testing data(0.30) in the seed 45 earlier on. 
- The grid was created to hold the hyperparameters and then CrossValidator() function was called using regression_pipe, grid, evaluator, and 3 as numfolds were used respectively. 
- Then the all_models was created using fit() on the training data.

- The script used for Feature Engineering and modeling was [ML.py](https://github.com/Pupat3l/NYC-trips/blob/main/mL_pyspark.py)

#### Results:
- Here are the results of the prediction (result of predictor) and the evaluations:
![img 1](https://github.com/Pupat3l/NYC-trips/blob/main/images/result1.png)
![img 2](https://github.com/Pupat3l/NYC-trips/blob/main/images/result2.png)

### Data Visualization:
- Utilized pandas, numpy, seaborn, matplotlib libraries to create visualizations.
- The visualization created were below:

![viz 1](https://github.com/Pupat3l/NYC-trips/blob/main/images/viz1.png)
![viz 1](https://github.com/Pupat3l/NYC-trips/blob/main/images/viz2.png)
![viz 1](https://github.com/Pupat3l/NYC-trips/blob/main/images/viz3.png)
![viz 1](https://github.com/Pupat3l/NYC-trips/blob/main/images/viz4.png)

## Tools
- Data Storage: AWS S3
- Data Processing: Python Scripts on Databricks 
- Visualization: Python 

## Getting Started
- [Python](https://www.python.com/) installed
- [DataBricks] community version to utlizie free hadoop cluster.
- [Amazon S3] access
