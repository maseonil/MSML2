import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("Gym Experience Level Project")

mlflow.sklearn.autolog(log_models=False, log_datasets=False) 

df = pd.read_csv("gym_dataset_preprocessing.csv")
X = df.drop(columns=['Experience_Level'])
y = df['Experience_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
input_example = X_train[0:5]

with mlflow.start_run(run_name="Random Forest Modelling"):

    dataset_info = mlflow.data.from_pandas(df, source="gym_dataset_preprocessing.csv")
    mlflow.log_input(dataset_info, context="training")

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

    mlflow.sklearn.log_model(
        sk_model=rf, 
        artifact_path="best_model", 
        input_example=input_example,
        registered_model_name="Gym_Experience_RF_Model"
    )
