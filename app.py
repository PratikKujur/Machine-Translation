from fastapi import FastAPI
from keras.models import load_model
import numpy as np
import uvicorn
import pandas as pd
import tensorflow as tf
from machine_translation.pipeline.prediction_pipeline import Predict_text
from machine_translation.constants import *
from machine_translation.pipeline.train_pipeline import Training_pipeline

app=FastAPI()

best_model=load_model('artifacts/machineTranslation_1.h5')

Tp=Training_pipeline()

train_data,test_data,val_data=Tp.start_data_ingestion(Data_Path=Data_Path)
eng_tokenizer=Tp.start_text_preprocessing(task='tokenization',data=train_data,lang_key=source,num_word=None)
hin_tokenizer=Tp.start_text_preprocessing(task='tokenization',data=train_data,lang_key=target,num_word=None)

@app.post("/predict")
async def predict_route(msg):
    
    Pt=Predict_text(input_text=msg, model=best_model, en_tokenizer=eng_tokenizer, hi_tokenizer=hin_tokenizer, max_seq_len=max_seq_len)
    tranlated_text=Pt.predict_translation()
    return tranlated_text

if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)