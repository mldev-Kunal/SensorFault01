import sys
from typing import Dict, Tuple
import os
import pandas as pd
import pickle
import yaml
import boto3
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


from src.constant import *
from src.exception import CustomException
from src.logger import logging


class MainUtils:
    def __init__(self) -> None:
        pass


    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)


        except Exception as e:
            raise CustomException(e, sys) from e


    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))


            return schema_config


        except Exception as e:
            raise CustomException(e, sys) from e


   


    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        logging.info("Entered the save_object method of MainUtils class")


        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)


            logging.info("Exited the save_object method of MainUtils class")


        except Exception as e:
            raise CustomException(e, sys) from e


   


    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info("Entered the load_object method of MainUtils class")


        try:
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)


            logging.info("Exited the load_object method of MainUtils class")


            return obj


        except Exception as e:
            raise CustomException(e, sys) from e
   
    @staticmethod    
    def load_object(file_path):
        try:
            with open(file_path,'rb') as file_obj:
                return pickle.load(file_obj)
        except Exception as e:
            logging.info('Exception Occured in load_object function utils')
            raise CustomException(e,sys)
         
    @staticmethod
    def evaluate_models(X_train, y_train, X_test, y_test, models, param_grid):
        try:
            report = {}

            for i in range(len(list(models))):
                model_name = list(models.keys())[i]
                logging.info(f"Starting evaluation for model: {model_name}")
                model = list(models.values())[i]
                para=param_grid[model_name]["search_param_grid"]

                logging.info(f"Performing GridSearchCV for {model_name}")
                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)

                logging.info(f"Best params for {model_name}: {gs.best_params_}")
                model.set_params(**gs.best_params_)
                
                logging.info(f"Retraining {model_name} with best params")
                model.fit(X_train,y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)
                
                logging.info(f"Model {model_name} scores - Train: {train_model_score}, Test: {test_model_score}")

                report[model_name] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)
