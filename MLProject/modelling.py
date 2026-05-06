import pandas as pd
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Mengatur lokasi file secara dinamis agar tidak FileNotFoundError di GitHub
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "gym_dataset_preprocessing.csv")

# Memuat dataset
df = pd.read_csv(csv_path)
X = df.drop(columns=['Experience_Level'])
y = df['Experience_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mlflow.sklearn.autolog()

with mlflow.start_run():
    rf = RandomForestClassifier(
        n_estimators=300, 
        max_depth=10, 
        random_state=42,
        n_jobs=-1 
    )
    rf.fit(X_train, y_train)

    accuracy = rf.score(X_test, y_test)
     mlflow.log_metric("accuracy", accuracy)
    
    # Log model ke folder bernama 'model'
    mlflow.sklearn.log_model(
        sk_model=rf, 
        artifact_path="model", 
        input_example=X_train[0:5]
    )
