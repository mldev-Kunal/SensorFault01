from flask import Flask, render_template, jsonify, request, send_file, redirect
from src.exception import CustomException
from src.logger import logging as lg
import os, sys

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return redirect("/predict")

@app.route("/train")
def train_route():
    try:
        train_pipeline = TrainingPipeline()
        run_pipeline_result = train_pipeline.run_pipeline()
        return f"Training Completed. Model Score: {run_pipeline_result}"
    except Exception as e:
        raise CustomException(e, sys)

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    try:
        if request.method == 'POST':
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_detail = prediction_pipeline.run_pipeline()
            lg.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                            download_name=prediction_file_detail.prediction_output_filename,
                            as_attachment=True)
        else:
            return render_template('upload_file.html')
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
