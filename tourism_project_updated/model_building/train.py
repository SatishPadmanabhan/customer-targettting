import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os
import joblib
from sklearn.utils.class_weight import compute_class_weight

# Define constants for the dataset path on Hugging Face
hf_dataset_repo_id = "satishp0879/tourism-customer-dataset"
hf_model_repo_id = "satishp0879/tourism-customer-model"

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Function to download data from Hugging Face
def download_from_hf(filename):
    local_path = f"./{filename}"
    print(f"Attempting to download {filename} from {hf_dataset_repo_id}...")
    api.hf_hub_download(
        repo_id=hf_dataset_repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir="./",
        local_dir_use_symlinks=False
    )
    print(f"Successfully downloaded {filename}.")
    return local_path

print("Downloading preprocessed data from Hugging Face...")
Xtrain_path = download_from_hf("Xtrain.csv")
Xtest_path = download_from_hf("Xtest.csv")
ytrain_path = download_from_hf("ytrain.csv")
ytest_path = download_from_hf("ytest.csv")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)
print("Data downloaded and loaded successfully.")

# Define numeric and categorical features (must match prep.py)
numeric_features = [
    'Age',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'PitchSatisfactionScore',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]
categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Passport',
    'OwnCar',
    'Designation'
]

# Preprocessing for XGBoost (One-Hot Encoding for categorical features)
def preprocess_data(X_df, cat_cols):
    # Ensure all categorical columns are treated as objects before one-hot encoding
    for col in cat_cols:
        if col in X_df.columns:
            X_df[col] = X_df[col].astype(str)

    # Apply one-hot encoding only to the specified categorical features
    X_processed = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)
    return X_processed

Xtrain_processed = preprocess_data(Xtrain, categorical_features)
Xtest_processed = preprocess_data(Xtest, categorical_features)

# Align columns - essential if train/test sets have different dummy variables (e.g., if a category is missing in one set)
train_cols = Xtrain_processed.columns
test_cols = Xtest_processed.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    Xtest_processed[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    Xtrain_processed[c] = 0

Xtest_processed = Xtest_processed[train_cols]

# Calculate class weights for imbalanced datasets
classes = np.unique(ytrain.values.ravel())
weights = compute_class_weight(class_weight='balanced', classes=classes, y=ytrain.values.ravel())
class_weights_dict = {c: w for c, w in zip(classes, weights)}

# For binary classification with XGBoost, often scale_pos_weight is used
# scale_pos_weight = count(negative examples) / count(positive examples)
negative_count = (ytrain == 0).sum().sum()
positive_count = (ytrain == 1).sum().sum()
scale_pos_weight_value = negative_count / positive_count if positive_count > 0 else 1


# Set MLflow tracking URI from environment variable (for production)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Tourism_Customer_Targetting_Model_Training_Production")

with mlflow.start_run():
    # Log fixed parameters for the grid search setup
    mlflow.log_param("model_type", "XGBoostClassifier_GridSearch")
    random_state = 42
    mlflow.log_param("random_state_for_xgb", random_state)
    mlflow.log_param("scale_pos_weight_fixed", scale_pos_weight_value)

    # Define the XGBoost Classifier with fixed parameters not in grid search
    xgb_clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=random_state,
        scale_pos_weight=scale_pos_weight_value
    )

    # Create a pipeline with the XGBoost Classifier (no preprocessing as it's done before)
    pipeline = Pipeline([
        ('xgbclassifier', xgb_clf)
    ])

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'xgbclassifier__n_estimators': [50, 75, 100],
        'xgbclassifier__max_depth': [2, 3, 4],
        'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
        'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
        'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
        'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6]
    }

    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1', # F1-score is often good for imbalanced datasets
        cv=5, # 5-fold cross-validation
        verbose=1,
        n_jobs=-1 # Use all available cores
    )

    grid_search.fit(Xtrain_processed, ytrain.values.ravel())

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("GridSearchCV completed.")
    print("Best parameters found: ", best_params)
    print("Best F1-score found (cross-validation): ", grid_search.best_score_)

    # Log best parameters from grid search
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    mlflow.log_metric("best_grid_search_f1_score_cv", grid_search.best_score_)

    # Make predictions with the best model
    y_pred = best_model.predict(Xtest_processed)

    # Evaluate the best model on the test set
    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)

    # Log test set metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)

    print(f"Model Trained and Evaluated (Best GridSearchCV Model on Test Set):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Log the best model
    mlflow.sklearn.log_model(best_model, "xgboost_gridsearch_model")
    print("Best model logged to MLflow.")

     # Save the model locally
    model_path = "Tourism_customer_targetting_best_model_V1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    model_upload_repo_id = hf_model_repo_id # Use the new model repo ID
    model_upload_repo_type = "model"

    # Step 1: Check if the model repository exists
    try:
        api.repo_info(repo_id=model_upload_repo_id, repo_type=model_upload_repo_type)
        print(f"Model repository '{model_upload_repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repository '{model_upload_repo_id}' not found. Creating new model repository...")
        create_repo(repo_id=model_upload_repo_id, repo_type=model_upload_repo_type, private=False)
        print(f"Model repository '{model_upload_repo_id}' created.")

    api.upload_file(
        path_or_fileobj="Tourism_customer_targetting_best_model_V1.joblib",
        path_in_repo="Tourism_customer_targetting_best_model_V1.joblib",
        repo_id=model_upload_repo_id,
        repo_type=model_upload_repo_type,
    )
