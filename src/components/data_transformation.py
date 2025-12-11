import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    artifact_dir = os.path.join(artifact_folder, "data_transformation")
    transformed_train_file_path = os.path.join(artifact_dir, "train.npy")
    transformed_test_file_path = os.path.join(artifact_dir, "test.npy")
    preprocessor_file_path = os.path.join(artifact_dir, "preprocessor.pkl")


class DataTransformation:
    def __init__(self, feature_store_file_path):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig()

        self.utils = MainUtils()
        
    @staticmethod
    def get_data(feature_store_file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(feature_store_file_path)
            data.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace = True)
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        try:
            imputer_step = ("imputer", SimpleImputer(strategy="constant", fill_value=0))
            scaler_step = ("scaler", RobustScaler())

            preprocessor = Pipeline(steps=[imputer_step, scaler_step])
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self):
        logging.info("Entered initiate data transformation method of data transfomration class")
        try:
            dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)
            logging.info("Got the data from feature store file path")
            X = dataframe.drop(TARGET_COLUMN, axis=1)
            y = np.where(dataframe[TARGET_COLUMN] == -1, 0, 1)
            logging.info("Got the features and target column")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info("Got the train and test data")

            preprocessor = self.get_data_transformer_object()

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            
            self.utils.save_object(file_path=preprocessor_path, object=preprocessor)

            train_array = np.c_[X_train_scaled, np.array(y_train)]
            test_array = np.c_[X_test_scaled, np.array(y_test)]

            return (
                train_array,
                test_array,
                preprocessor_path
            )

        except Exception as e:
            raise CustomException(e, sys) from e



            
            
            

            

            

    


