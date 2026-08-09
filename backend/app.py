from flask import Flask, render_template, request
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score


app = Flask(__name__, template_folder="../frontend", static_folder="../frontend")

@app.route("/")
def index():
    return render_template("ELA.html")

@app.route("/train", methods=["POST"])
def receive_settings():
    uploaded_file = request.files["file"]
    uploaded_file.save("data.csv")    
    settings_text = request.form.get("settings")
    data = json.loads(settings_text) if settings_text else {}
    
    with open("settings.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    with open("settings.json", "r", encoding="utf-8") as f:
        settings = json.load(f)
        selected_feature = settings["fe"]
        data = pd.read_csv("data.csv")
        data = data.dropna()
        data = data.drop_duplicates()
        x = data[selected_feature]
        x = x.select_dtypes(include=['number'])
        selected_target = settings["ta"]
        y = data[selected_target].loc[x.index]
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(x_train,y_train)
        pred = model.predict(x_test)
        print(m1(y_test,pred))

    return {"status": "success"}
def m1(yt,yp):
    RMSE =  root_mean_squared_error(y_true=yt,y_pred=yp)
    R2 = r2_score(y_true=yt,y_pred=yp)
    return RMSE, R2
    
