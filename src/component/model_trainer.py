from catboost import CatBoostClassifier
from sklearn.ensemble import(
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor, 
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

from dataclasses import dataclass
import os
import sys
@dataclass
class ModelTrainderConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainderConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # from the data transformation we get the train and the test array 
            logging.info("Splitting training and testing input data")
            # from the array we split the value to the x and y train and test 
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LogisticRegression(max_iter=1000),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostClassifier(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            # give all the model to the evaluate models function in the utils file to find the best model name 
            # It will return a dictionary with the model name and the score
            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
            