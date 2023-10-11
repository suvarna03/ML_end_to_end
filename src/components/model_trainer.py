import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    "criterion": ["mse", "friedman_mse", "mae", "poisson"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 10, 20, 30],
                },
                "Random Forest":{
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "criterion": ["mse", "mae"],
                    "max_features": ["auto", "sqrt", "log2", None],
                },
                "Gradient Boosting":{
                    "loss": ["ls", "lad", "huber", "quantile"],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "max_features": ["auto", "sqrt", "log2"],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # Initialize best score and name
            best_model_score = -1  # Choose an initial value lower than the possible scores
            best_model_name = None
            best_model = None

            for model_name, model in models.items():
            # ... your model training and evaluation code

            # Calculate the test score for the current model
                current_test_model_score = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)

            # Update the best score and name if a better model is found
                if current_test_model_score > best_model_score:
                    best_model_score = current_test_model_score
                    best_model_name = model_name
                    best_model = model

                # After the loop, check if a valid best model is found
            if best_model_score >= 0.6:
                logging.info(f"Best model found: {best_model_name} with R-squared score {best_model_score}")
            else:
                raise CustomException("No best model found")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square            
        except Exception as e:
            raise CustomException(e,sys)