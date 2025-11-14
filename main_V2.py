from fastapi import FastAPI, HTTPException, status, File, UploadFile

from io import StringIO
import pandas as pd
from joblib import load

# from models import Prediction, Base
from datetime import datetime
import pytz
import os
from google.cloud import bigquery


app = FastAPI()



@app.post("/predict")
async def predict_houseprice():

    classifier = load("linear_regression.joblib")
    
    # Leer los features desde BigQuery
    client = bigquery.Client()
    query = """
        SELECT string_field_0 as feature_name
        FROM `<PROJECT_ID>.<DATASET>.selected_features`
    """
    query_job = client.query(query)
    feature_rows = query_job.result()
    features = [row.feature_name for row in feature_rows]


    query_prediction = """
        SELECT *
        FROM `<PROJECT_ID>.<DATASET>.xtrain`
    """

    df = client.query(query_prediction).to_dataframe()
    df = df[features]
    
    predictions = classifier.predict(df)

    lima_tz = pytz.timezone('America/Lima')
    now = datetime.now(lima_tz)
    
    predictions_df = pd.DataFrame({
        'file_name': 'test',
        'prediction': predictions,
        'created_at': now
    })
    
    predictions_df.to_gbq(
        destination_table='<DATASET>.<TABLE>', 
        project_id='<PROJECT_ID>',
        if_exists='append'
    )

    return {
        "predictions": predictions.tolist()
    }