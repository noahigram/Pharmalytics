import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
df_drug = pd.read_excel("Healthcare_dataset.xlsx", 'Dataset')
df_drug1 = df_drug.drop(['Gender', 'Race', 'Region', 'Ethnicity', 'Age_Bucket', 'Risk_Immobilization', 'Risk_Estrogen_Deficiency',
                        'Risk_Chronic_Liver_Disease', 'Risk_Untreated_Early_Menopause', 'Risk_Untreated_Chronic_Hyperthyroidism', 'Risk_Osteogenesis_Imperfecta'], axis=1)


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get input data
        to_predict_list = request.form.to_dict()

        # Call preprocessDataAndPredict function and pass inputs
        try:
            prediction = preprocessDataAndPredict(to_predict_list)

            # Pass prediction to prediction template
            return render_template('/predict.html', prediction=prediction)

        except ValueError as e:
            return str(e)  # "Please Enter valid values" UNCOMMENT ME

        pass
    pass


def preprocessDataAndPredict(feature_dict):

    # Convert input to dataframe
    test_data = {k: [v] for k, v in feature_dict.items()}

    # Need to add other features- web app only asks users for gender, race, ethnicity, region and age bucket
    for feat in df_drug1.columns:
        test_data[feat] = df_drug[feat][1]

    print(test_data)

    other_features = []
    test_data = pd.DataFrame(test_data)

    # Open trained model
    file = open("model.pkl", "rb")

    # Load trained model
    trained_model = pickle.load(file)

    # Get prediction results
    predict = trained_model.predict(test_data)

    return predict

    pass


if __name__ == "__main__":
    app.run(debug=True)
