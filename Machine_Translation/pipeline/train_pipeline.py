import pandas as pd
from machine_translation.constants import *
from machine_translation.components.data_ingestion import Dataingestion
from machine_translation.components.data_preprocessing import Data_Preprocessing
from machine_translation.components.get_model import Get_Model
from machine_translation.components.model_trainer import Model_Trainer

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
            special_token = text_preprocessing_tokenizer.preprocess_with_special_tokens(data,en_tokenizer, hi_tokenizer, max_seq_len)
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
     
    def model_creation(self,model_type,**kwargs):
        def encoder(input_dim,output_dim,hidden_units):
            model=Get_Model()
            encoder_model=model.encoder(input_dim,output_dim,hidden_units)
            return encoder_model
        

        def decoder(input_dim,output_dim,hidden_units,encoder_state_h, encoder_state_c):
            model=Get_Model()
            decoder_model=model.decoder(input_dim,output_dim,hidden_units,encoder_state_h, encoder_state_c)
            return decoder_model
        

        def attention(decoder_outputs, encoder_outputs):
            model=Get_Model()
            attention_model=model.attention(decoder_outputs, encoder_outputs)
            return attention_model
        

        def get_final_model(encoder_inputs, decoder_inputs,context_vector,vocab_size_hi,decoder_outputs):
            model=Get_Model()
            final_model=model.get_final_model(encoder_inputs, decoder_inputs,context_vector,vocab_size_hi,decoder_outputs)
            return final_model
        

        if model_type == "encoder":
            return encoder(
                kwargs["input_dim"], kwargs["output_dim"], kwargs["hidden_units"]
            )
        elif model_type == "decoder":
            return decoder(
                kwargs["input_dim"], kwargs["output_dim"], kwargs["hidden_units"], kwargs["encoder_state_h"], kwargs["encoder_state_c"]
            )
        elif model_type == "attention":
            return attention(
                kwargs["decoder_outputs"],kwargs["encoder_outputs"]
            )
        elif model_type == "get_final_model":
            return get_final_model(
                kwargs["encoder_inputs"],kwargs["decoder_inputs"], kwargs["context_vector"],kwargs["vocab_size_hi"],kwargs["decoder_outputs"]
            )
        else:
            raise ValueError("Invalid mode. Choose 'encoder' or 'decoder' or 'attention' or 'get_final_model'")

    def model_trianing(self,model,en_train, dec_train_input,dec_train_target,en_val, dec_val_input,dec_val_target):
        model_trainer=Model_Trainer()
        model_history=model_trainer.start_model_training(model,en_train, dec_train_input,dec_train_target,en_val, dec_val_input,dec_val_target)
        return model_history

