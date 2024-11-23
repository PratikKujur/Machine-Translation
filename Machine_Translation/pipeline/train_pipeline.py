import pandas as pd
from machine_translation.constants import *
from machine_translation.components.data_ingestion import Dataingestion
from machine_translation.components.data_preprocessing import Data_Preprocessing


class Training_pipeline:
    def start_data_ingestion(self,Data_Path):
        self.Data_Path=Data_Path
        data=Dataingestion(self.Data_Path)
        train_data,test_data,val_data=data.read_data()
    
        return train_data,test_data,val_data

    def start_text_preprocessing(self,task,**kwargs):
        def tokenization_processing(data, lang_key, num_word):
            text_preprocessing_tokenizer = Data_Preprocessing()
            tokenizer = text_preprocessing_tokenizer.tokenization(data, lang_key, num_word)
            return tokenizer
        def special_token_processing(data,en_tokenizer, hi_tokenizer, max_seq_len):
            text_preprocessing_tokenizer = Data_Preprocessing()
            special_token = text_preprocessing_tokenizer.preprocess_with_special_tokens(
                data,en_tokenizer, hi_tokenizer, max_seq_len
            )
            return special_token
        if task == "tokenization":
            return tokenization_processing(
                kwargs["data"], kwargs["lang_key"], kwargs.get("num_word", None)
            )
        elif task == "special_tokens":
            return special_token_processing(
                kwargs["data"],kwargs["en_tokenizer"], kwargs["hi_tokenizer"], kwargs["max_seq_len"]
            )
        else:
            raise ValueError("Invalid mode. Choose 'tokenization' or 'special_tokens'.")
        

        

        