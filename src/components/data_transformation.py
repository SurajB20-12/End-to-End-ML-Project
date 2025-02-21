import os
import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import Coustom_exception

from dataclasses import dataclass

from src.utils import save_object


@dataclass
class DataTransformationConfig:
  processor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
  def __init__(self):
    self.data_transformation_config=DataTransformationConfig()

  def get_data_transformer_object(self):

    try:
      numerical_features=['writing score','reading score']

      categorical_features=['gender', 'race/ethnicity', 
                            'parental level of education', 'lunch', 'test preparation course']
      
      numerical_pipeline=Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy="median")),
          ("scaler",StandardScaler())
        ]
      )

      categorical_pipeline=Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy="most_frequent")),
          ('one-hot-encoder',OneHotEncoder()),
          ("scaler",StandardScaler(with_mean=False))
        ]
      )

      logging.info(f'Numerical Columns:{numerical_features}')

      logging.info(f'Categorical Columns:{categorical_features}')

      preprocessor=ColumnTransformer([
        ("numerical_pipeline",numerical_pipeline,numerical_features),
        ("categorical_pipeline",categorical_pipeline,categorical_features)
      ])

      return preprocessor
    except Exception as e:
      raise Coustom_exception(e,sys)
    
  def initiate_data_transformation(self,train_path,test_path):
    try:
      train_df=pd.read_csv(train_path)
      test_df=pd.read_csv(test_path)

      logging.info("Read train and test data completed")

      logging.info("obtaining processing obj")

      processing_obj=self.get_data_transformer_object()

      # accessing target column name 
      target_col_name="math score"

      '''
      drop target feature from train data 
      so we apply transformation on independent 
      feature in train data
      '''
      input_feature_train_df=train_df.drop(columns=[target_col_name],axis=1)
      target_feature_train_df=train_df[target_col_name]

      '''
      drop target feature from test data 
      so we apply transformation on independent 
      feature in test data
      '''

      input_feature_test_df=test_df.drop(columns=[target_col_name],axis=1)
      target_feature_test_df=test_df[target_col_name]

      logging.info("Applying processing obj on training and testing dataframe")

      
      input_feature_train_arr=processing_obj.fit_transform(input_feature_train_df)

      input_feature_test_arr=processing_obj.transform(input_feature_test_df)

      train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]

      test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

      logging.info('saved processing object')

      # creating save_object() function in utils.py 

      save_object(file_path=self.data_transformation_config.processor_obj_file_path,obj=processing_obj)

      return(
        train_arr,
        test_arr,
        self.data_transformation_config.processor_obj_file_path,
      )

    except Exception as e:
      raise Coustom_exception(e,sys)