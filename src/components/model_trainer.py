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

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

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

      params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "iterations": [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
            }

      model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

      best_model_name = max(model_report, key=lambda k: model_report[k]["R2 Score"])
      best_model = models[best_model_name]
      best_score = model_report[best_model_name]["R2 Score"]

      

      if best_score<0.6:
        raise Coustom_exception("No best model found")
      logging.info(f"Best found model on both training and testing dataset {best_model_name} with R2 Score: {best_score}")
      
      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )

      # Get test set predictions

      y_pred_test = best_model.predict(X_test)

      # Calculate final evaluation metrics

      final_r2 = r2_score(y_test, y_pred_test)
      final_mae = mean_absolute_error(y_test, y_pred_test)
      final_mse = mean_squared_error(y_test, y_pred_test)
      final_rmse = final_mse ** 0.5
     

      return {
        "Best Model": best_model_name,
        "R2 Score": final_r2,
        "MAE": final_mae,
        "MSE": final_mse,
        "RMSE": final_rmse,
        
      }
    except Exception as e:
      raise Coustom_exception(e,sys)