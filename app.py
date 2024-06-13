from flask import Flask,request,jsonify,render_template 
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

app  = Flask(__name__)


## import pkl file----------------->>>>>>>>>
reg_model = pickle.load(open("Model-file/reg.pkl","rb"))
standard_scaler = pickle.load(open("Model-file/scaler.pkl","rb"))


## Route for home page ---------->>>>>>>>>>>
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
         sepal_length = float(request.form.get("sepal_length"))
         sepal_width= float(request.form.get("sepal_width"))
         petal_length = float(request.form.get("petal_length"))
         petal_width = float(request.form.get("petal_width"))
         
         
         
         

         new_data_scaled = standard_scaler.transform([[sepal_length,sepal_width,petal_length,petal_width]])
         result = reg_model.predict(new_data_scaled)

         return render_template("home.html",result=result[0])

    else:
        return render_template("home.html")


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)

