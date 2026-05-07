import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    n_estimators      = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    max_depth         = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    min_samples_split = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    criterion         = sys.argv[4] if len(sys.argv) > 4 else "entropy"
    
    default_csv = os.path.join(base_dir, "gym_dataset_preprocessing.csv")
    file_path = sys.argv[5] if len(sys.argv) > 5 else default_csv
    
    data = pd.read_csv(file_path)

    X = data.drop("Experience_Level", axis=1)
    y = data["Experience_Level"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    input_example = X_train[0:5]

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            n_jobs=-1, 
            random_state=42
        )
        
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        ) 
