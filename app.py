import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from flask_cors import cross_origin


app = Flask(__name__)
app.static_folder = 'static'

with open("standardScalar.sav", 'rb') as f:
    scalar = pickle.load(f)

with open("modelForPrediction.sav", 'rb') as f:
    model = pickle.load(f)

with open("pca_model.sav", 'rb') as f:
    pca_model = pickle.load(f)


@app.route("/", methods=["POST", "GET"])
@cross_origin()
def homepage():
    return render_template("homepage.html")


@app.route("/predict", methods=["POST", "GET"])
@cross_origin()
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    data_df = pd.DataFrame(final_features, index=[1, ])
    scaled_data = scalar.transform(data_df)
    principal_data = pca_model.transform(scaled_data)
    prediction = model.predict(principal_data)

    return render_template("report.html", prediction = prediction[0])



if __name__ == "__main__":
    app.run()

