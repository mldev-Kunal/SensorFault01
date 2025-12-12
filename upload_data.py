from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd
import json

#url
load_dotenv()
uri = os.getenv("MONGODB_URL")

# Create a new client and connect to the server
client = MongoClient(uri)

#create database name and collection name
DATABASE_NAME="ML"
COLLECTION_NAME='Wafer_fault'

df = pd.read_csv(r'C:\Users\kp224\Downloads\SensorFaultagy\notebooks\wafer_23012020_041211.csv')

df.drop("Unnamed: 0", axis = 1, inplace = True)

json_record = list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)