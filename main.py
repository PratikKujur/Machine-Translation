import pandas as pd
from machine_translation.constants import *
from machine_translation.components.data_ingestion import Dataingestion
from machine_translation.pipeline.train_pipeline import Training_pipeline

Tp=Training_pipeline()

# step 1- start data ingestion (read parquet file and perform split of Data in train and test and val from train)
train_data,test_data,val_data=Tp.start_data_ingestion(Data_Path=Data_Path)
print(train_data.shape,test_data.shape,val_data.shape)


