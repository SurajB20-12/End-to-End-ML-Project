import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd 

from src.pipeline.predict_pipline import CustomData,PredictPipline
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)


@application.route('/')
def index():
  return render_template('index.html')

@application.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
  if request.method =='GET':
    return render_template('home.html')
  else:
    data=CustomData(gender=request.form.get('gender'),
      ethnicity=request.form.get('ethnicity'),
      parental_level_of_education=request.form.get('parental_level_of_education'),
      lunch=request.form.get('lunch'),
      test_preparation_course=request.form.get('test_preparation_course'),
      reading_score=float(request.form.get('reading_score')),
      writing_score=float(request.form.get('writing_score'))
    )
    dataframe=data.get_data_as_data_frame()

    print(dataframe)

    predict_pipeline=PredictPipline()
    results=predict_pipeline.predict(dataframe)

    print("Prediction Result:", results) 

    return render_template('home.html',result=results[0])
  
if __name__=='__main__':
  application.run(host="0.0.0.0",port=5000,debug=True)