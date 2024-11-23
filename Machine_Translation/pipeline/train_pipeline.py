import pandas as pd
from machine_translation.constants import *
from machine_translation.components.data_ingestion import Dataingestion


class Training_pipeline:
    def start_data_ingestion(self,Data_Path):
        self.Data_Path=Data_Path
        data=Dataingestion(self.Data_Path)
        train_data,test_data,val_data=data.read_data()
    
        return train_data,test_data,val_data

        
        