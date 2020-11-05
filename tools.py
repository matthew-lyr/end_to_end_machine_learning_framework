from google.cloud import bigquery
from google.oauth2 import service_account
import pandas_gbq
import pandas as pd
from pandas.io import gbq
import numpy as np
from IPython import display
import tensorflow as tf
from tensorflow.python.data import Dataset
import math
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from googleapiclient import discovery
from googleapiclient import errors
import time
import re
import os
from google.cloud import storage
import datetime
import pytz



class Auto_predictor():
    def __init__(self, service_account_credential_file, project_id, dataset_id, model_name, bucket, sql, region = "us-east1", data_format  = 'JSON',  max_worker_count=8,  version_name=None,runtime_version=None):

        timestamp   = datetime.datetime.now(pytz.timezone("America/New_York")).strftime('%Y_%m_%d_%H_%M_%S')
        current_day = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
        last_hour   = datetime.datetime.now(pytz.timezone('US/Eastern')).replace(microsecond=0,second=0,minute=0).strftime("%H:%M:%S")
        
        self.credentials = service_account.Credentials.from_service_account_file(service_account_credential_file)
        self.project_id  = project_id
        self.dataset_id  = dataset_id
        self.model_name  = model_name
        self.bucket      = bucket
        self.sql         = sql

        self.region           = region
        self.data_format      = data_format
        self.version_name     = version_name
        self.max_worker_count = max_worker_count
        self.runtime_version  = runtime_version
        self.job_id           = '{}_{}'.format(model_name,timestamp)

        self.input_bucket  = '{}/prediction_input/{}'.format(bucket,model_name)
        self.output_bucket = '{}/prediction_output/{}'.format(bucket,model_name)
       
    def __repr__(self):
        return ('model_name:{}\ninput_bucket:{}\noutput_bucket:{}\nsql:{}'.format(self.model_name,self.input_bucket,self.output_bucket,self.sql))
    
    def prepare_input_data(self):
        # This function takes data from bigQuery, massages it or do any necessary feature engineering in a SQL query, and dumps the result into a GCS bucket
        # At the time of writing, the functionality of directly dumping bigQuery query result into a GCS bucket is not supported,
        # So I used a temp table for storing the query result before dumping it to the GCS bucket.

        client                       = bigquery.Client(credentials=self.credentials,project=self.project_id)
        job_config                   = bigquery.QueryJobConfig()
        table_ref                    = client.dataset(self.dataset_id).table("TEMP_TABLE")
        job_config.destination       = table_ref
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        query_job                    = client.query(self.sql,location="US",job_config=job_config,)  
        query_job.result()    
        print("Saved query result to {}".format(table_ref.path))


        job_config                    = bigquery.job.ExtractJobConfig()
        job_config.destination_format = bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON
        gcs_destination               = '{}/*.{}'.format(self.bucket,self.data_format.lower())
        extract_job                   = client.extract_table(table_ref, gcs_destination, job_config = job_config, location='US') 
        extract_job.result() 
        print('Dumped {}:{}.{} to {}'.format(self.project_id, self.dataset_id, "TEMP_TABLE", self.bucket))

            
    def batch_predict(self):
        # Google Cloud has specific naming requirements for project and model id 
        project_id = 'projects/{}'.format(self.project_id)
        model_id   = 'projects/{}/models/{}'.format(self.project_id, self.model_name)

        body = {'jobId'          : self.job_id,
                'predictionInput': {'dataFormat' : self.data_format,
                                    'inputPaths' : self.input_bucket,
                                    'outputPath' : self.output_bucket,
                                    'region'     : self.region,
                                    'versionName': self.version_name,
                                    'modelName'  : model_id}}
        if self.max_worker_count:
            body['predictionInput']['maxWorkerCount'] = self.max_worker_count

        if self.runtime_version:
            body['predictionInput']['runtimeVersion'] = self.runtime_version

        # Make batch prediction

        ml      = discovery.build('ml', 'v1', credentials=self.credentials)
        request = ml.projects().jobs().create(parent=project_id, body=body)
        request.execute()

class Trainer():
    def __init__(self, project_id, service_account_dir):
        self.project_id = project_id
        self.service_account_dir = service_account_dir
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.service_account_dir

    
    def forward_key_to_export(self,estimator):
        estimator = tf.contrib.estimator.forward_features(estimator, self.passed_through_id_fields)
        config = estimator.config
        def model_fn2(features, labels, mode):
            estimatorSpec = estimator._call_model_fn(features, labels, mode, config=config)
            if estimatorSpec.export_outputs:
                for ekey in self.passed_through_id_fields:
                    estimatorSpec.export_outputs[ekey] = tf.estimator.export.PredictOutput(estimatorSpec.predictions)
            return estimatorSpec
        return tf.estimator.Estimator(model_fn=model_fn2, config=config)

    def auto_serving_fn(self):
        def adding_passing_through_ids():
            fields = self.training_features.columns
            INSTANCE_KEY_COLUMN = [field for field in fields if field.lower() in self.passed_through_id_fields]
            INPUT_COLUMNS = set(tf.feature_column.numeric_column(field,dtype=tf.dtypes.float64) if field.lower() in self.passed_through_id_fields else tf.feature_column.numeric_column(field) for field in fields)
            inputs = {}
            features = {}
            for feat in INPUT_COLUMNS:
                inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)
                if feat.name in INSTANCE_KEY_COLUMN:
                    features[feat.name] = tf.identity(inputs[feat.name])
                else:
                    features[feat.name] = inputs[feat.name]
            serving_input_rcvr = tf.estimator.export.ServingInputReceiver(features, inputs)
            return serving_input_rcvr
        serving_fn = adding_passing_through_ids
        return serving_fn()


    def construct_feature_columns(self, input_features):
        input_features_except_passthru_keys = input_features.copy().drop(self.passed_through_id_fields,axis = 1)
        return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features_except_passthru_keys])

    def my_input_fn(self, features, targets, batch_size=1, shuffle=True, num_epochs=None, shuffle_num = 10000):
        features = {key:np.array(value) for key,value in dict(features).items()}
        ds = Dataset.from_tensor_slices((features,targets)) 
        ds = ds.batch(batch_size).repeat(num_epochs)
        if shuffle:
            ds = ds.shuffle(shuffle_num)
        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels

    def predict_training_input_fn(self):
        return my_input_fn(self.training_features, self.training_targets[self.target_name], num_epochs=1, shuffle=False)

    def predict_validation_input_fn(self):
        return my_input_fn(self.validation_features, self.validation_targets[self.target_name], num_epochs=1, shuffle=False)

    def train(self, sql, steps, dropout_num, batch_size, hidden_units, target_name, gcs_bucket_name = "model_files", tmp_model_prefix = "temporary_models/auto_train", completed_model_prefix = "completed_models/auto_train", is_classification = True, learning_rate = 0.0003,decay_rate = 0.9,clip_gradients_by_norm = 5.0,passed_through_id_fields = []):
        self.is_classification        = is_classification
        self.training_data            = gbq.read_gbq(sql, self.project_id, dialect='standard')
        self.target_name              = target_name
        self.gcs_bucket_name          = self.project_id + "_" + gcs_bucket_name
        self.tmp_model_prefix         = tmp_model_prefix
        self.completed_model_prefix   = completed_model_prefix
        self.training_features        = self.training_data.loc[:, self.training_data.columns != target_name]
        self.training_targets         = self.training_data.loc[:, self.training_data.columns == target_name]
      # self.tmp_model_dir            = "gs://{}/{}".format(gcs_bucket_name,tmp_model_prefix)
        self.completed_model_dir      = "gs://{}/{}".format(self.gcs_bucket_name,completed_model_prefix)
        self.passed_through_id_fields = passed_through_id_fields

        # Create an estimator.
        optimizer     = tf.contrib.estimator.clip_gradients_by_norm(tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay_rate),clip_gradients_by_norm)

        if is_classification:
            estimator = tf.estimator.DNNClassifier(
              feature_columns = self.construct_feature_columns(self.training_features),
              hidden_units    = hidden_units,
              optimizer       = optimizer,
              dropout         = dropout_num
              #,
              #model_dir       = self.tmp_model_dir
            )
        else:
            estimator = tf.estimator.DNNRegressor(
              feature_columns = self.construct_feature_columns(self.training_features),
              hidden_units    = hidden_units,
              optimizer       = optimizer,
              dropout         = dropout_num
              #,
              #model_dir       = self.tmp_model_dir
            )

        estimator         = self.forward_key_to_export(estimator)
        training_input_fn = lambda: self.my_input_fn(self.training_features, self.training_targets[target_name], batch_size=batch_size)
        estimator.train(input_fn=training_input_fn, steps=steps)
        self.estimator    = estimator
        print("training has been completed")



    def export_model(self):
        storage_client        = storage.Client(project=self.project_id)
        try:
            bucket_name = storage_client.create_bucket(self.gcs_bucket_name)
        except:
            bucket_name = storage_client.get_bucket(self.gcs_bucket_name)
        completed_model_files = bucket_name.list_blobs(prefix=self.completed_model_dir)
        for file in completed_model_files:
            file.delete()

        self.completed_model_dir = self.estimator.export_savedmodel(export_dir_base = self.completed_model_dir, serving_input_receiver_fn = self.auto_serving_fn)
        self.completed_model_dir = re.sub("'","",re.sub("b'", "", str(self.completed_model_dir)))
        # temp_model_files         = bucket_name.list_blobs(prefix=self.tmp_model_prefix)
        # for file in temp_model_files:
        #     file.delete()
        print("Model has been exported to {}".format(self.completed_model_dir))
        return self.completed_model_dir



    def validate(self, validation_data_sql):
        self.validation_data     = gbq.read_gbq(validation_data_sql, self.project_id, dialect='standard')
        self.validation_features = self.validation_data[:, self.validation_data.columns != self.target_name]
        self.validation_targets  = self.validation_data[:, self.validation_data.columns == self.target_name]

        training_predictions   = self.estimator.predict(input_fn=self.predict_training_input_fn)
        validation_predictions = self.estimator.predict(input_fn=self.predict_validation_input_fn)


        if self.is_classification:
            training_predictions            = [item for item in training_predictions]
            training_predictions            = pd.concat([pd.DataFrame(np.array([item['class_ids'][0] for item in training_predictions]))], axis=1)
            training_predictions.columns    = ['class_ids','classes','logistic','logits','probabilities']
            training_predictions['label']   = pd.DataFrame(np.array(self.training_targets[self.target_name]))

            validation_predictions          = [item for item in validation_predictions]
            validation_predictions          = pd.concat([pd.DataFrame(np.array([item['class_ids'][0] for item in validation_predictions]))], axis=1)
            validation_predictions.columns  = ['class_ids','classes','logistic','logits','probabilities']
            validation_predictions['label'] = pd.DataFrame(np.array(self.validation_targets[self.target_name]))

            def weird_division(n, d):
                        return n / d if d else 0

            training_true_positive = training_predictions[(training_predictions['label']==1)&(training_predictions['class_ids']==1)].shape[0]
            training_true_negative = training_predictions[(training_predictions['label']==0)&(training_predictions['class_ids']==0)].shape[0]
            training_real          = training_predictions[(training_predictions['label']==1)].shape[0]
            training_positive      = training_predictions[(training_predictions['class_ids']==1)].shape[0]
            training_all           = training_predictions.shape[0]
            
            training_accuracy       = weird_division((training_true_positive + training_true_negative),training_all)
            training_real_recall    = weird_division((training_true_positive),training_real)
            training_real_precision = weird_division((training_true_positive),training_positive)
            training_fake_recall    = weird_division((training_true_negative),(training_all-training_real))
            training_fake_precision = weird_division((training_true_negative),(training_all-training_positive))
            
            validation_true_positive = validation_predictions[(validation_predictions['label']==1)&(validation_predictions['class_ids']==1)].shape[0]
            validation_true_negative = validation_predictions[(validation_predictions['label']==0)&(validation_predictions['class_ids']==0)].shape[0]
            validation_real          = validation_predictions[(validation_predictions['label']==1)].shape[0]
            validation_positive      = validation_predictions[(validation_predictions['class_ids']==1)].shape[0]
            validation_all           = validation_predictions.shape[0]
            
            validation_accuracy       = weird_division((validation_true_positive + validation_true_negative),validation_all)
            validation_real_recall    = weird_division((validation_true_positive),validation_real)
            validation_real_precision = weird_division((validation_true_positive),validation_positive)
            validation_fake_recall    = weird_division((validation_true_negative),(validation_all-validation_real))
            validation_fake_precision = weird_division((validation_true_negative),(validation_all-validation_positive))

            print(
            'training_accuracy:{}\nvalidation_accuracy:{}\ntraining_real_recall:{}\nvalidation_real_recall:{}\ntraining_real_precision:{}\nvalidation_real_precision:{}\ntraining_fake_recall:{}\nvalidation_fake_recall:{validation_fake_recall}\ntraining_fake_precision:{training_fake_precision}\nvalidation_fake_precision:{validation_fake_precision}\n'.format(
            training_accuracy, validation_accuracy, training_real_recall, validation_real_recall, training_real_precision, validation_real_precision, training_fake_recall, validation_fake_recall, training_fake_precision, validation_fake_precision))   
        else:
            training_predictions               = np.array([item['predictions'][0] for item in training_predictions])
            validation_predictions             = np.array([item['predictions'][0] for item in validation_predictions])
            training_root_mean_squared_error   = math.sqrt(metrics.mean_squared_error(training_predictions, self.training_targets))
            validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_predictions, self.validation_targets))
            print("training_root_mean_squared_error:{}, validation_root_mean_squared_error:{}".format(training_root_mean_squared_error, validation_root_mean_squared_error))

class Model_manager():
    def __init__(self, modelName, versionName, gcp_project_id, trainedModelLocation):
        self.modelName            = modelName
        self.versionName          = versionName
        self.gcp_project_id       = gcp_project_id
        self.trainedModelLocation = trainedModelLocation

    def remove_model(self):
        project_id = 'projects/{}'.format(self.gcp_project_id)
        model_id   = '{}/models/{}'.format(project_id, self.modelName)
        ml         = discovery.build('ml','v1',cache_discovery=False)
        request    = ml.projects().models().versions().list(parent=model_id)
        try:
            response   = request.execute()
        except:
            return 'Model not found'


        # check if model has versions
        if 'versions' in response.keys():
            versions = response['versions']
            while len(versions) >= 1:
                for version in response['versions']:
                    request = ml.projects().models().versions().delete(name = version['name'])
                    try:
                        request.execute()
                    except errors.HttpError as err:
                        reason = err._get_reason()
                        if 'Cannot delete the default version' in reason:
                            next

                request = ml.projects().models().versions().list(parent=model_id)
                response = request.execute()
                time.sleep(1)
                try:
                    versions = response['versions']
                except:
                    break

        # remove the model
        request = ml.projects().models().delete(name=model_id)

        # make the call to remove
        while True:
            try:
                response = request.execute()
                # Any additional code on success goes here (logging, etc.)
            except errors.HttpError as err:
                # Something went wrong, print out some information.
                print('There was an error deleting the model.' +
                      ' Check the details:')
                reason = err._get_reason()
                print(reason)
                # Wait for 1000 milliseconds.
                if 'A model with versions cannot be deleted' in reason:
                    # this probably means that the last delete version call
                    # has not yet completed, so wait and retry
                    time.sleep(1)
                    continue
            break

    def create_model(self, versionDescription = '', runtimeVersion = '1.13', pythonVersion = '3.5', machineType = 'mls1-c1-m2'):
        ml          = discovery.build('ml', 'v1', cache_discovery=False)
        projectID   = 'projects/{}'.format(self.gcp_project_id)
        requestDict = {'name': self.modelName, 'description': 'Another model for testing.'}
        request     = ml.projects().models().create(parent=projectID, body=requestDict)
        try:
            response = request.execute()
        except errors.HttpError as err:
            print('There was an error creating the model.' +  ' Check the details:')
            print(err._get_reason())
        response = None


        modelID = '{}/models/{}'.format(projectID, self.modelName)
        requestDict = {
          'name': self.versionName,
          'description': versionDescription,
          'deploymentUri': self.trainedModelLocation,
          'runtimeVersion': runtimeVersion,
          'pythonVersion': pythonVersion,
          'machineType': machineType}
        request = ml.projects().models().versions().create(parent=modelID, body=requestDict)
        try:
            response = request.execute()
            operationID = response['name']
        except errors.HttpError as err:
            print('There was an error creating the version.' + ' Check the details:')
            print(err._get_reason())
