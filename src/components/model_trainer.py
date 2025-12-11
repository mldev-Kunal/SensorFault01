import sys
from typing import Generator, List, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_dir = os.path.join(artifact_folder)
    trained_model_path = os.path.join(artifact_dir, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join("config", "model.yaml")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.models = {
            "XGBClassifier": XGBClassifier(),
            "SVC": SVC(),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier()
        }


    def model_trainer(self, X, y, models):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            report = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]

                model.fit(X_train, y_train) # Train model

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

                return report

        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model(self, X_train:np.array,
                    y_train:np.array,
                    X_test:np.array,
                    y_test:np.array):

        try:
            model_report: dict = self.evaluate_models(X_train, y_train, X_test, y_test, models=self.models)

            print(model_report)

            best_model_score = max(sorted(model_report.values(), reverse=True))

            ## To get the model name of the best model from the dictionary

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model_object = self.models[best_model_name]

            return best_model_name, best_model_object, best_model_score

        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(self, X_train,
                            y_train,
                            best_model_name,
                            best_model_object,
                            best_model_name)->object:
        try:

            model_params_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"][best_model_name]["search_param_grid"]

            grid_search = GridSearchCV(best_model_object, model_params_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            
            best_params = grid_search.best_params_
            
            print("best params are: ", best_params)

            fine_tuned_model = best_model_object.set_params(**best_params)
            
            return fine_tuned_model

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Splitting training and testing input and target feature")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info(f"Extracting model config file path")

            model_report: dict = self.evaluate_models(X_train, y_train, models=self.models)
            
            ## To get best model score from dict

            best_model_score = max(sorted(model_report.values()))
            ## To get best model name from dict

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = self.models[best_model_name]

            best_model = self.finetune_best_model(X_train, y_train, best_model_name, best_model)

            best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            best_model_train_score = accuracy_score(y_train, y_train_pred)
            best_model_test_score = accuracy_score(y_test, y_test_pred)

            print(f"Best model train score: {best_model_train_score}")
            print(f"Best model test score: {best_model_test_score}")

            if best_model_test_score < self.model_trainer_config.expected_accuracy:
                raise CustomException("No best model found for expected accuracy")

            logging.info(f"Saving best model: {self.model_trainer_config.trained_model_path}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            self.utils.save_object(file_path=self.model_trainer_config.trained_model_path, obj=best_model)

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)

