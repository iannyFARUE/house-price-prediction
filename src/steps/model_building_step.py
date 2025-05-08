import logging
from typing import Annotated, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step
from zenml.client import Client

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model

model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="best_model_pipeline", is_model_artifact=True)]:
    """
    Builds and trains multiple regression models, compares them, and returns the best performer.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # Define preprocessing for categorical and numerical features
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Define different models to train and compare
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    best_model_name = None
    best_score = float('-inf')  # For R2, higher is better

    # Check if there's an active run and use it instead of starting a new one
    active_run = mlflow.active_run()
    if active_run:
        run_id = active_run.info.run_id
        logging.info(f"Using active MLflow run: {run_id}")
    else:
        logging.info("No active MLflow run found, not starting a new one as ZenML likely manages this")
    
    # Enable autologging
    mlflow.sklearn.autolog()
    
    # Create, train and evaluate each model
    for model_name, model_instance in models.items():
        try:
            logging.info(f"Training {model_name}...")
            
            # Create a pipeline with the preprocessor and the current model
            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", model_instance)]
            )
            
            # Log model name as a parameter
            mlflow.log_param(f"model_type_{model_name}", model_name)
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Evaluate with cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=5, scoring='r2'
            )
            mean_cv_score = np.mean(cv_scores)
            
            # Log additional metrics
            y_pred = pipeline.predict(X_train)
            mse = mean_squared_error(y_train, y_pred)
            mae = mean_absolute_error(y_train, y_pred)
            r2 = r2_score(y_train, y_pred)
            
            # Log metrics to MLflow with model name prefix to distinguish between models
            mlflow.log_metric(f"{model_name}_mean_cv_r2", mean_cv_score)
            mlflow.log_metric(f"{model_name}_mse", mse)
            mlflow.log_metric(f"{model_name}_mae", mae)
            mlflow.log_metric(f"{model_name}_r2", r2)
            
            # Store results
            results[model_name] = {
                "pipeline": pipeline,
                "cv_r2": mean_cv_score,
                "mse": mse,
                "mae": mae,
                "r2": r2
            }
            
            # Update best model if this one is better
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_model_name = model_name
                
            logging.info(f"{model_name} - CV R2: {mean_cv_score:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
        except Exception as e:
            logging.error(f"Error training {model_name}: {e}")
    
    # Log the best model details
    if best_model_name:
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_model_cv_r2", best_score)
        logging.info(f"Best model: {best_model_name} with CV R2 score: {best_score:.4f}")
        
        # Create comparison table and log as artifact
        comparison_data = {
            "Model": [],
            "CV R2": [],
            "MSE": [],
            "MAE": [],
            "R2": []
        }
        
        for model_name, info in results.items():
            comparison_data["Model"].append(model_name)
            comparison_data["CV R2"].append(info["cv_r2"])
            comparison_data["MSE"].append(info["mse"])
            comparison_data["MAE"].append(info["mae"])
            comparison_data["R2"].append(info["r2"])
        
        comparison_table = pd.DataFrame(comparison_data)
        comparison_path = "model_comparison.csv"
        comparison_table.to_csv(comparison_path, index=False)
        mlflow.log_artifact(comparison_path)
        
        logging.info(f"Model comparison:\n{comparison_table}")

    # Don't end the MLflow run as ZenML is managing it
    # Return the best model pipeline
    return results[best_model_name]["pipeline"] if best_model_name else None