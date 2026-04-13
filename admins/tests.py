import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

file_path = r"C:\Users\tarun\OneDrive\Desktop\anusha (2)\anusha\media\final_dataset_30000.csv"
df = pd.read_csv(file_path)

# Encode Gender
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

# One hot encode symptoms
symptom_cols = ['Symptom1','Symptom2','Symptom3','Symptom4','Symptom5','Symptom6','Symptom7']
df = pd.get_dummies(df, columns=symptom_cols)

# Encode Disease
le = LabelEncoder()
df['Disease'] = le.fit_transform(df['Disease'])

# Features & Target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# Model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))