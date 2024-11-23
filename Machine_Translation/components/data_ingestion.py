import pandas as pd
from machine_translation.constants import *
from sklearn.model_selection import train_test_split


class Dataingestion:
    def __init__(self,data_path):
        self.Data_Path=data_path
        
    def read_data(self):
        data=pd.read_parquet(self.Data_Path,engine=engine)
        train_data,test_data=train_test_split(data,test_size=test_size,random_state=random_state)
        train_data,val_data=train_test_split(train_data,test_size=val_size,random_state=random_state)
        return train_data,test_data,val_data
    
