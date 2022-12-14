import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

class Preprocess:
    def __init__(self):
        self.data_path = "data/train.csv"
        self.test_size = 0.2
        self.numword = 1000
        self.maxlen = 20

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df.drop(["id","keyword","location"],axis=1)

        #define data and label to train
        train = df['text'].values
        label = df['target'].values

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(train, label, test_size=self.test_size)

    def Tokenization(self):
        self.tokens = Tokenizer(num_words=self.numword)
        self.tokens.fit_on_texts(self.X_train)

    def sequence_to_token(self, input):
        sequence = self.tokens.texts_to_sequences(input)
        return pad_sequences(sequence, maxlen=self.maxlen)        