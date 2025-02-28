import os
import sys

import pandas as pd
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.exception import Coustom_exception
def save_object(file_path,obj):
  try:
    dir_path=os.path.dirname(file_path)

    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,'wb') as file_obj:
      pickle.dump(obj,file_obj)

  except Exception as e:
    raise Coustom_exception(e,sys)


def evaluate_model(X_train,X_test,y_train,y_test,models,params):
  try:
    report={}

    for i in range(len(list(models))):
      model= list(models.values())[i]
      para=params[list(models.keys())[i]]
      #model.fit(X_train,y_train) #training each models from models dict 

      gs=GridSearchCV(model,para,cv=3,scoring="r2",error_score="raise")
      gs.fit(X_train,y_train)

      model.set_params(**gs.best_params_)
      model.fit(X_train,y_train)

      y_pred_train=model.predict(X_train) #predicting train result

      y_pred_test=model.predict(X_test) #predicting test result

      train_model_score=r2_score(y_train,y_pred_train) #train model score

      test_model_score=r2_score(y_test,y_pred_test) #test model score

      mae = mean_absolute_error(y_test, y_pred_test)
      mse = mean_squared_error(y_test, y_pred_test)
      rmse = mse ** 0.5  # Root Mean Squared Error
      

      report[list(models.keys())[i]]={
        "R2 Score": test_model_score,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
      }
    return report
  except Exception as e:
    raise Coustom_exception(e,sys)
  
def load_object(file_path):
  try:
    with open(file_path,'rb') as file_obj:
      return pickle.load(file_obj)
  except Exception as e:
    raise Coustom_exception