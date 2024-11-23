from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from machine_translation.constants import *

class Data_Preprocessing:
    def __init__(self):
        pass
    
    def tokenization(self,data,lang_key,num_words=None):
        tokenizer = Tokenizer(num_words=num_words, filters=filters, lower=lower, oov_token=oov_token)
        tokenizer.fit_on_texts(special_tokens + data[lang_key].tolist())
        return tokenizer
    
    
    def preprocess_with_special_tokens(self,data, en_tokenizer, hi_tokenizer, max_seq_len):
        en_sequences = en_tokenizer.texts_to_sequences(data[source])
        hi_sequences = [start_token + sent + end_token for sent in data[target]]
        hi_sequences = hi_tokenizer.texts_to_sequences(hi_sequences)

        en_sequences = pad_sequences(en_sequences, maxlen=max_seq_len, padding=post_padding)
        hi_sequences = pad_sequences(hi_sequences, maxlen=max_seq_len + 2, padding=post_padding)

        decoder_input = hi_sequences[:, :-1]
        decoder_target = hi_sequences[:, 1:]

        return en_sequences, decoder_input, decoder_target