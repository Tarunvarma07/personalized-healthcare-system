import pandas as pd
import numpy as np
import joblib

def predict_disease(age, gender, symptoms):

    # Load saved objects
    model = joblib.load("media/best_model.pkl")
    scaler = joblib.load("media/scaler.pkl")
    disease_encoder = joblib.load("media/disease_encoder.pkl")
    gender_encoder = joblib.load("media/gender_encoder.pkl")
    feature_columns = joblib.load("media/feature_columns.pkl")

    # =========================
    # CREATE INPUT DATAFRAME
    # =========================
    data = {
        "Age": age,
        "Gender": gender
    }

    for i in range(7):
        data[f"Symptom{i+1}"] = symptoms[i]

    input_df = pd.DataFrame([data])

    # Encode gender
    input_df["Gender"] = gender_encoder.transform(input_df["Gender"])

    # OneHot encode symptoms
    symptom_cols = [f"Symptom{i}" for i in range(1, 8)]
    input_df = pd.get_dummies(input_df, columns=symptom_cols)

    # =========================
    # IMPORTANT: MATCH TRAIN FEATURES
    # =========================
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)

    return disease_encoder.inverse_transform(pred)[0]