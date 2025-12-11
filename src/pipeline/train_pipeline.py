import sys
import os

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, feature_store_file_path):
        try:
            data_transformation = DataTransformation(feature_store_file_path=feature_store_file_path)
            train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation()
            return train_array, test_array, preprocessor_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, train_array, test_array):
        try:
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(train_array, test_array)
            return model_score
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_array, test_array, preprocessor_path = self.start_data_transformation(feature_store_file_path)
            r2_score = self.start_model_training(train_array, test_array)
            return r2_score

            print(f"Training completed. Trained model with r2 score: {r2_score}")

        except Exception as e:
            raise CustomException(e, sys)

