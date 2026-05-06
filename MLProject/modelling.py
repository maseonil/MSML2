import pandas as pd
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd

# Mendapatkan path absolut ke direktori tempat script ini berada
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "gym_dataset_preprocessing.csv")

# Memuat dataset menggunakan path yang sudah digabung
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    # Fallback jika file berada di root (untuk keamanan)
    df = pd.read_csv("gym_dataset_preprocessing.csv")

# 1. Tentukan Eksperimen (Sesuai dengan yang ada di YAML)
mlflow.set_experiment("gym_exp_project")

# 2. Matikan autolog untuk model agar kita bisa simpan manual dengan nama folder yang benar
mlflow.sklearn.autolog(log_models=False) 

# Memuat dataset
df = pd.read_csv(csv_path)
X = df.drop(columns=['Experience_Level'])
y = df['Experience_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
input_example = X_train[0:5]

with mlflow.start_run(run_name="Random Forest Modelling"):
    
    rf = RandomForestClassifier(
        n_estimators=300, 
        max_depth=10, 
        min_samples_split=5, 
        criterion='entropy', 
        random_state=42,
        n_jobs=-1 
    )
    
    rf.fit(X_train, y_train)
    
    test_accuracy = rf.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # 3. Simpan Model Secara Manual
    # artifact_path="model" adalah kunci agar Build Docker di YAML tidak error
    mlflow.sklearn.log_model(
        sk_model=rf, 
        artifact_path="model", 
        input_example=input_example
    )
