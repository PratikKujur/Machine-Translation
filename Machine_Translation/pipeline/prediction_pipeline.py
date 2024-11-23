import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Predict_text:
    def __init__(self,input_text, model, en_tokenizer, hi_tokenizer, max_seq_len):
        self.input_text=input_text
        self.model=model
        self.en_tokenizer=en_tokenizer
        self.hi_tokenizer=hi_tokenizer
        self.max_seq_len=max_seq_len


    def predict_translation(self):
        input_seq = self.en_tokenizer.texts_to_sequences([self.input_text])
        input_seq = pad_sequences(input_seq, maxlen=self.max_seq_len, padding="post")

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.hi_tokenizer.word_index['<sos>']

        decoded_sentence = []

        for _ in range(self.max_seq_len + 2):
            predictions = self.model.predict([input_seq, target_seq], verbose=0)
            predicted_token = np.argmax(predictions[0, -1, :])

            sampled_word = self.hi_tokenizer.index_word.get(predicted_token, '<unk>')
            if sampled_word == '<eos>':
                break

            decoded_sentence.append(sampled_word)

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = predicted_token

        return ' '.join(decoded_sentence)