import logging
import os
from datetime import datetime

LOG_FILE_NAME = f"log_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"

log_path = os.path.join(os.getcwd(), "logs", LOG_FILE_NAME)

os.makedirs(log_path, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
    filemode='w'
)

