import logging
import os
from datetime import datetime

LOG_FILE_NAME = f"log_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"

logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

log_path = os.path.join(logs_dir, LOG_FILE_NAME)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
    filemode='w'
)

