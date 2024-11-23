import pandas as pd
import tensorflow as tf
from machine_translation.constants import *
from machine_translation.components.data_ingestion import Dataingestion
from machine_translation.pipeline.train_pipeline import Training_pipeline
from machine_translation.pipeline.prediction_pipeline import Predict_text

Tp=Training_pipeline()

# step 1- Start data ingestion (read parquet file and perform split of Data in train and test and val from train)
train_data,test_data,val_data=Tp.start_data_ingestion(Data_Path=Data_Path)
print(train_data.shape,test_data.shape,val_data.shape)

#step 2- Text preprocessing- 2.1-Tokenization + 2.2-Handling special tokens
#step 2.1-

eng_tokenizer=Tp.start_text_preprocessing(task='tokenization',data=train_data,lang_key=source,num_word=None)
hin_tokenizer=Tp.start_text_preprocessing(task='tokenization',data=train_data,lang_key=target,num_word=None)
"""
#step 2.2-
en_train, dec_train_input, dec_train_target = Tp.start_text_preprocessing(task='special_tokens',
                                                                          data=train_data,
                                                                          en_tokenizer=eng_tokenizer,
                                                                          hi_tokenizer=hin_tokenizer,
                                                                          max_seq_len=max_seq_len)

en_val, dec_val_input, dec_val_target = Tp.start_text_preprocessing(task='special_tokens',
                                                                    data=test_data,
                                                                    en_tokenizer=eng_tokenizer,
                                                                    hi_tokenizer=hin_tokenizer,
                                                                    max_seq_len=max_seq_len)
en_test, _, _ = Tp.start_text_preprocessing(task='special_tokens',
                                            data=val_data,
                                            en_tokenizer=eng_tokenizer,
                                            hi_tokenizer=hin_tokenizer,
                                            max_seq_len=max_seq_len)

print(en_train.shape,dec_train_input.shape, dec_train_target.shape)

vocab_size_en = len(eng_tokenizer.word_index) + 1
vocab_size_hi = len(hin_tokenizer.word_index) + 1
#step-3 Get 3.1-lstm encoder,3.2-lstm decoder,3.3-attention,3.4-final model

#step 3.1-
encoder_inputs,encoder_outputs, encoder_state_h, encoder_state_c=Tp.model_creation(model_type='encoder',
                                                                                   input_dim=vocab_size_en,
                                                                                   output_dim=embedding_dim,
                                                                                   hidden_units=hidden_units)



#step 3.2-
decoder_inputs,decoder_outputs=Tp.model_creation(model_type='decoder',input_dim=vocab_size_en,
                                                 output_dim=embedding_dim,
                                                 hidden_units=hidden_units,
                                                 encoder_state_h=encoder_state_h,
                                                 encoder_state_c=encoder_state_c)
print(decoder_outputs.shape)
#step 3.3-
context_vector=Tp.model_creation(model_type='attention',
                                 decoder_outputs=decoder_outputs, 
                                 encoder_outputs=encoder_outputs)

print(context_vector.shape)
#step 3.4-
model=Tp.model_creation(model_type='get_final_model',
                        encoder_inputs=encoder_inputs, 
                        decoder_inputs=decoder_inputs,
                        context_vector=context_vector,
                        vocab_size_hi=vocab_size_hi,
                        decoder_outputs=decoder_outputs)
print(model.summary())
#step-4 model trainer
model_history=Tp.model_trianing(model,en_train, dec_train_input,dec_train_target,en_val, dec_val_input,dec_val_target)
print(model_history)
#step-5-model evaluation
"""
#step-model prediction
best_model=tf.keras.models.load_model('artifacts/machineTranslation_1.h5')
Pt=Predict_text(input_text=text,model=best_model, en_tokenizer=eng_tokenizer, hi_tokenizer=hin_tokenizer, max_seq_len=max_seq_len)

print(Pt.predict_translation())



