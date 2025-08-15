import os
import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# Models
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Custom utilities
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, param_grids):
        """
        Runs GridSearchCV for each model, trains with best parameters, 
        and returns a report of R2 scores.
        """
        model_report = {}
        best_models = {}

        for name, model in models.items():
            logging.info(f"Training {name} with hyperparameter tuning...")
            param_grid = param_grids.get(name, {})

            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='r2')
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                # If no param grid, train directly
                model.fit(X_train, y_train)
                best_model = model

            # Predict and score
            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)

            model_report[name] = score
            best_models[name] = best_model

            logging.info(f"{name} R2 Score: {score:.4f}")

        return model_report, best_models

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Initiating model training...")

            # Split into features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "KNeighbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0)
            }

            # Hyperparameters
            param_grids = {
                "Random Forest": {
                    "n_estimators": [50, 100, 300],
                    "max_depth": [None, 5, 15, 30],
                    "min_samples_split": [2, 4, 6],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "min_samples_split": [2, 5]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 300],
                    "learning_rate": [0.01, 0.1, 1.0],
                    "algorithm": ["SAMME", "SAMME.R"]
                },
                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },
                "KNeighbors": {
                    "n_neighbors": [3, 5, 11],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 15],
                    "min_samples_split": [2, 4, 6]
                },
                "XGBoost": {
                    "n_estimators": [50, 100, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 6, 10],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_lambda": [1, 1.5, 2.0]
                },
                "CatBoost": {
                    "iterations": [100, 300, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 10],
                    "l2_leaf_reg": [1, 3, 5]
                }
            }

            # Evaluate all models
            model_report, best_models = self.evaluate_models(
                X_train, y_train, X_test, y_test, models, param_grids
            )

            # Get best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = best_models[best_model_name]

            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score:.4f}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException("Error in initiate_model_trainer method", sys) from e


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation

    # Paths to training & testing CSV files
    train_csv_path = "artifacts/train.csv"
    test_csv_path = "artifacts/test.csv"

    # Step 1: Data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_path=train_csv_path,
        test_path=test_csv_path
    )

    # Step 2: Train model
    trainer = ModelTrainer()
    best_model_name, best_score = trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Best Model: {best_model_name} with R2 Score: {best_score:.4f}")
