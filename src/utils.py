import os
import sys

import pandas as pd
import pickle
from sklearn.metrics import r2_score
from src.exception import Coustom_exception
def save_object(file_path,obj):
  try:
    dir_path=os.path.dirname(file_path)

    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,'wb') as file_obj:
      pickle.dump(obj,file_obj)

  except Exception as e:
    raise Coustom_exception(e,sys)


def evaluate_model(X_train,X_test,y_train,y_test,models):
  try:
    report={}

    for i in range(len(list(models))):
      model= list(models.values())[i]
      model.fit(X_train,y_train) #training each models from models dict 

      y_pred_train=model.predict(X_train) #predicting train result

      y_pred_test=model.predict(X_test) #predicting test result

      train_model_score=r2_score(y_train,y_pred_train) #train model score

      test_model_score=r2_score(y_test,y_pred_test) #test model score

      report[list(models.keys())[i]]=test_model_score
    return report
  except Exception as e:
    raise Coustom_exception(e,sys)