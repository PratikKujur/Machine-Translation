from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

class Get_Model:
    def encoder(self,vocab_size_en,embedding_dim,hidden_units):
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(input_dim=vocab_size_en, output_dim=embedding_dim)(encoder_inputs)
        encoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding)
        return encoder_inputs,encoder_outputs, encoder_state_h, encoder_state_c

    def decoder(self,vocab_size_hi,embedding_dim,hidden_units,encoder_state_h, encoder_state_c):
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(input_dim=vocab_size_hi, output_dim=embedding_dim)(decoder_inputs)
        decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[encoder_state_h, encoder_state_c])
        return decoder_inputs,decoder_outputs
    
    def attention(self,decoder_outputs, encoder_outputs):
        attention_layer = Attention()
        context_vector = attention_layer([decoder_outputs, encoder_outputs])
        return context_vector
    
    def get_final_model(self,encoder_inputs, decoder_inputs,context_vector,vocab_size_hi,decoder_outputs):
        decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_outputs])
        decoder_dense = Dense(vocab_size_hi, activation="softmax")
        decoder_outputs = decoder_dense(decoder_combined_context)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model