import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.ensemble import(
  AdaBoostRegressor,
  GradientBoostingRegressor,
  RandomForestRegressor
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import Coustom_exception
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
  trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()

  def initiate_model_training(self,train_data,test_data):
    try:
      logging.info('Split the traing and test input')
      X_train,y_train,X_test,y_test=(
        train_data[:,:-1],
        train_data[:,-1],
        test_data[:,:-1],
        test_data[:,-1]
      )

      models={
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Linear Regression": LinearRegression(),
        "XGBRegressor": XGBRegressor(),
        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        "AdaBoost Regressor": AdaBoostRegressor(),
      }

      model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

      best_score=max(sorted(model_report.values()))

      best_model_name=list(model_report.keys())[list(model_report.values()).index(best_score)]

      best_model=models[best_model_name]

      if best_score<0.6:
        raise Coustom_exception("No best model found")
      logging.info(f"Best found model on both training and testing dataset {best_model_name}")
      
      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )

      prdiction=best_model.predict(X_test)
      R2_score=r2_score(y_test,prdiction)

      return(
        R2_score
      )
    except Exception as e:
      raise Coustom_exception(e,sys)