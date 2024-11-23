from fastapi import FastAPI
from keras.models import load_model
import numpy as np
import uvicorn
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer



app=FastAPI()

df=pd.read_parquet("english-to-hindi/data/train-00000-of-00001-71c2cec7402cd444.parquet",engine="pyarrow")
model=load_model('machineTranslation.h5')
max_seq_len=20


def tokenize_with_special_tokens(data, lang_key, num_words=None):
    tokenizer = Tokenizer(num_words=num_words, filters='', lower=True, oov_token="<unk>")
    special_tokens = ["<sos>", "<eos>"]
    tokenizer.fit_on_texts(special_tokens + data[lang_key].tolist())
    return tokenizer

en_tokenizer = tokenize_with_special_tokens(df, "english_sentence")
hi_tokenizer = tokenize_with_special_tokens(df, "hindi_sentence")

def predict_translation(input_text, model, en_tokenizer, hi_tokenizer, max_seq_len):
    
    input_seq = en_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_len, padding="post")

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = hi_tokenizer.word_index['<sos>']

    decoded_sentence = []

    for _ in range(max_seq_len + 2):
        predictions = model.predict([input_seq, target_seq], verbose=0)
        predicted_token = np.argmax(predictions[0, -1, :])

        sampled_word = hi_tokenizer.index_word.get(predicted_token, '<unk>')
        if sampled_word == '<eos>':
            break

        decoded_sentence.append(sampled_word)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = predicted_token

    return ' '.join(decoded_sentence)

@app.post("/predict")
async def predict_route(text):
    tranlated_text=predict_translation(text, model, en_tokenizer, hi_tokenizer, max_seq_len)
    return tranlated_text

if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
