import sys
import os
import pandas as pd
from src.exception import Coustom_exception
from src.utils import load_object

class PredictPipline:
  def __init__(self):
    pass

  def predict(self,features):
    model_path=os.path.join('artifacts','model.pkl')
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
    model=load_object(file_path=model_path)
    preprocessor=load_object(file_path=preprocessor_path)

    data_scaled=preprocessor.transform(features)
    pred=model.predict(data_scaled)

    return pred
class CustomData:
  def __init__(  self,
        gender: str,
        ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.ethnicity = ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score
  def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise Coustom_exception(e, sys)