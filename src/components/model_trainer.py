import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exeption import CustomException  # fixed spelling
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Initiating model training...")
            
            # Split into X and y
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

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

            model_report = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                model_report[name] = r2
                logging.info(f"{name} R2 Score: {r2}")

            # Get best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException("Error in initiate_model_trainer method", sys) from e

if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation

    # Paths to your training & testing CSV files
    train_csv_path = "artifacts/train.csv"
    test_csv_path = "artifacts/test.csv"


    # Step 1: Run data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_path=train_csv_path,
        test_path=test_csv_path
    )

    # Step 2: Train model
    trainer = ModelTrainer()
    best_model_name, best_score = trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Best Model: {best_model_name} with R2 Score: {best_score}")
