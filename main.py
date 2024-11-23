import pandas as pd
from machine_translation.constants import *
from machine_translation.components.data_ingestion import Dataingestion
from machine_translation.pipeline.train_pipeline import Training_pipeline

Tp=Training_pipeline()

# step 1- Start data ingestion (read parquet file and perform split of Data in train and test and val from train)
train_data,test_data,val_data=Tp.start_data_ingestion(Data_Path=Data_Path)
print(train_data.shape,test_data.shape,val_data.shape)

#step 2- Text preprocessing- 2.1-Tokenization + 2.2-Handling special tokens
#step 2.1-

eng_tokenizer=Tp.start_text_preprocessing(task='tokenization',data=train_data,lang_key=source,num_word=None)
hin_tokenizer=Tp.start_text_preprocessing(task='tokenization',data=train_data,lang_key=target,num_word=None)

#step 2.2-
en_train, dec_train_input, dec_train_target = Tp.start_text_preprocessing(task='special_tokens',data=train_data,en_tokenizer=eng_tokenizer,hi_tokenizer=hin_tokenizer,max_seq_len=max_seq_len)
en_val, dec_val_input, dec_val_target = Tp.start_text_preprocessing(task='special_tokens',data=test_data,en_tokenizer=eng_tokenizer,hi_tokenizer=hin_tokenizer,max_seq_len=max_seq_len)
en_test, _, _ = Tp.start_text_preprocessing(task='special_tokens',data=val_data,en_tokenizer=eng_tokenizer,hi_tokenizer=hin_tokenizer,max_seq_len=max_seq_len)

print(en_train.shape,dec_train_input.shape, dec_train_target.shape)