import numpy as np
from flask import Flask,request, jsonify,render_template
import pickle
import os
import pandas as pd
from pandas import DataFrame
########### creating the flask  app
pic_folder = os.path.join("Static","imgs")

flask_app = Flask(__name__)

flask_app.config['UPLOAD_FOLDER'] = pic_folder
model = pickle.load(open("covidmodel.pkl","rb"))

@flask_app.route("/")
def Home():
    #pic1 = os.path.join(flask_app.config['UPLOAD_FOLDER'],"1img.jpg")
    return render_template("home.html")#user_img =pic1)
@flask_app.route("/predict",methods =['POST'])
def predict(): # number = [x for x in request.form.values()] which has to be int
    number = [x for x in request.form.values()]
    max1 = pd.to_datetime(number[0], format= "%Y-%m-%d")
    min1 = pd.to_datetime('2021-05-17', format= "%Y-%m-%d")
    delta = max1-min1
    model_prediction= model.forecast(delta.days)
    model_prediction =model_prediction.astype(int)# float  to int conversion 
    a =np.array(model_prediction)
    a=a.tolist()
    result = a
    pic1 = os.path.join(flask_app.config['UPLOAD_FOLDER'],"2img.jpg")

    
    #df = pd.DataFrame(a.values,columns= ["Beds available"])
    return render_template ("home.html",abc= delta.days ,res= result,date_picked =number[0],prediction =result[-1],user_img =pic1)#prediction_text =a
if __name__ == "__main__":
    flask_app.run(debug=True)






