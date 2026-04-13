import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from django.conf import settings


def train_models():

    print("🔄 Starting FAST training pipeline...")

    # =======================
    # LOAD DATA
    # =======================
    file_path = os.path.join(settings.MEDIA_ROOT, 'final_dataset_30000.csv')

    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset not found")

    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df.dropna(inplace=True)

    os.makedirs("media", exist_ok=True)

    # =======================
    # ENCODING
    # =======================

    # Gender encoding
    gender_encoder = LabelEncoder()
    df['Gender'] = gender_encoder.fit_transform(df['Gender'])

    # Symptom OneHot encoding
    symptom_cols = [f'Symptom{i}' for i in range(1, 8)]
    df = pd.get_dummies(df, columns=symptom_cols)

    # Save feature columns
    feature_columns = df.drop('Disease', axis=1).columns
    joblib.dump(feature_columns, "media/feature_columns.pkl")

    # Disease encoding
    disease_encoder = LabelEncoder()
    df['Disease'] = disease_encoder.fit_transform(df['Disease'])

    joblib.dump(gender_encoder, "media/gender_encoder.pkl")
    joblib.dump(disease_encoder, "media/disease_encoder.pkl")

    # =======================
    # FEATURES
    # =======================
    X = df.drop('Disease', axis=1)
    y = df['Disease']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "media/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )

    # =======================
    # FAST MODELS
    # =======================
    base_models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=25),
        "RandomForest": RandomForestClassifier(
            n_estimators=120,
            max_depth=30,
            n_jobs=-1,
            random_state=42
        ),
        "NaiveBayes": GaussianNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=800,
            n_jobs=-1
        )
    }

    model_instances = []
    accuracies = {}
    best_model = None
    best_model_name = ""
    best_accuracy = 0

    # =======================
    # TRAIN MODELS
    # =======================
    for name, model in base_models.items():
        print(f"⚙️ Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        accuracies[name] = acc
        joblib.dump(model, f"media/{name}_model.pkl")

        model_instances.append((name, model))

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    # =======================
    # VOTING ENSEMBLE
    # =======================
    print("🤝 Training Voting Ensemble...")
    voting = VotingClassifier(
        estimators=model_instances,
        voting='soft'
    )

    voting.fit(X_train, y_train)

    y_vote = voting.predict(X_test)
    vote_acc = accuracy_score(y_test, y_vote)

    accuracies["Voting"] = vote_acc
    joblib.dump(voting, "media/Voting_model.pkl")

    if vote_acc > best_accuracy:
        best_model = voting
        best_model_name = "Voting"
        best_accuracy = vote_acc

    joblib.dump(best_model, "media/best_model.pkl")

    print("\n✅ FAST TRAINING COMPLETE")
    print("🏆 Best Model:", best_model_name)
    print("🎯 Accuracy:", best_accuracy)

    return accuracies, best_model_name