# End-to-end Machine Learning Framework on Google Cloud Platform

### Overview
This project provides a simply way to implement a scalable machine learning system on Google Cloud Platform that handles on-going data cleaning, model training, model updating, and prediction for large-scale structured data.

### Special Features
1. Able to make large-scale batch predictions for more than hundreds of millions of records. 
2. Data cleaning and preparation is simple and straight-forward: use SQL to describe how data needs to be prepared.
3. Trained models can be easily re-trained and updated periodically.

### How it works
![alt text](https://github.com/matthew-lyr/end_to_end_machine_learning_framework/blob/main/End_to_end_ml_high_level_flowchart.PNG
)
There are three main pieces in the framework:
1. Trainer grabs data from the data source and trains a model.
2. Model_manager deploys the trained mode or use it to replace a deployed model.
3. Auto_predictor then uses the deployed model to make batch predictions on the input data specified by users in SQL.

In essence, what the framework is really doing is to utilize various services provided by Google Cloud, such as BigQuery, Cloud Storage, and ML Engine/AI Platform. The framework utilizes their strengths and orchestrates them into a connected system. 

![alt text](https://github.com/matthew-lyr/end_to_end_machine_learning_framework/blob/main/End_to_end_ml_detailed_flowchart.PNG
)
