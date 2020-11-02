import keras
import pandas as pd
import gensim
import numpy as np
import string

seed_value = 101
np.random.seed(seed_value)

import random
random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)

from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import LambdaCallback
from keras.models import load_model

# import nltk
# from nltk import word_tokenize, pos_tag

#gpus = tf.config.experimental.list_physical_devices('GPU') # for cudnn
#tf.config.experimental.set_memory_growth(gpus[0], True)

import sys
sys.executable

# hyperparameter config
max_sentence_len=10
epoch_num = 5

# read the inspiring set 
# haiku-like dataset provided by the haikurnn 
data = pd.read_csv("./data/haikus.csv")
data.head()

sentences_x = data['0'].tolist()
sentences_y = data['1'].tolist()
sentences_y_p = data['2'].tolist()
print(len(sentences_x),len(sentences_y),len(sentences_y_p))

# preparing sentences_tokens list for word2vec training
# for word2vec training, I use all of the sentences of the poems
# therefore it only needs to add all poems in one list 
# sentences' order in one poem is remained
# sentences_x_tokens et al. are prepared for training the generator model

sentences_x_tokens = [[word for word in str(sentence).lower().translate(str.maketrans('','', string.punctuation)).split()[:max_sentence_len]] for sentence in sentences_x]
sentences_y_tokens = [[word for word in str(sentence).lower().translate(str.maketrans('','', string.punctuation)).split()[:max_sentence_len]] for sentence in sentences_y]
sentences_y_p_tokens = [[word for word in str(sentence).lower().translate(str.maketrans('','', string.punctuation)).split()[:max_sentence_len]] for sentence in sentences_y_p]

sentences_tokens = []
for i in range(len(sentences_x_tokens)): # add blank to avoid missing tokenizing in next step
  sentences_tokens.append(sentences_x_tokens[i] + sentences_y_tokens[i] + sentences_y_p_tokens[i])
print(sentences_tokens[:5])

# train the word2vec model, using default config
# note that in gensim.model.Word2vec, the default method is CBOW
# and negative sampling will be user, where negative = 5
print("Word2vec model training starts")
word_model = gensim.models.Word2Vec(sentences_tokens, size=50, min_count=1, window=5, iter=100)
pretrained_weights = word_model.wv.vectors
vocab_size, emdedding_size = pretrained_weights.shape
print("Word2vec model training ends")

def word2idx(word):
  return word_model.wv.vocab[word].index
def idx2word(idx):
  return word_model.wv.index2word[idx]

sentences_x_train = sentences_x_tokens + sentences_y_tokens
sentences_y_train = sentences_y_tokens + sentences_y_p_tokens
train_x = np.zeros([len(sentences_x_train), max_sentence_len], dtype=np.int32)
train_y = np.zeros([len(sentences_y_train)], dtype=np.int32)
for i, sentence in enumerate(sentences_x_train):
  for t, word in enumerate(sentence):
    train_x[i, t] = word2idx(word)
  if sentence!=[] and sentences_y_train[i]!=[]:
    train_y[i] = word2idx(sentences_y_train[i][0])

# build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=emdedding_size, name='x1'))
model.add(Dense(units=vocab_size, name='y1'))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# this temperature-tuned generating mechanism is a sample 
# from keras for generating rnn
# see "Character-level text generation with LSTM"
# link: https://keras.io/examples/generative/lstm_character_level_text_generation/

def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)
 
def generate_next(text, num_generated=10):
  word_idxs = [word2idx(word) for word in text.lower().split()]
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature=1.0)
    word_idxs.append(idx)
  return ' '.join(idx2word(idx) for idx in word_idxs)

vocab = word_model.wv.vocab.keys()
random_start = random.choice(tuple(vocab))
def poem_generator(start):
  first_sen = generate_next(start, num_generated=2)
  second_sen = generate_next(first_sen, num_generated=5)
  third_sen = generate_next(second_sen, num_generated=3)
  return [first_sen, " ".join(second_sen.split()[-5:]), " ".join(third_sen.split()[-3:])]

def callback_ep(epoch, _):
  random_start = random.choice(tuple(vocab))
  haikulike_ep = poem_generator(random_start)
  print('\r')
  print(haikulike_ep[0])
  print(haikulike_ep[1])
  print(haikulike_ep[2])

print("LSTM model training starts")
model.fit(train_x, train_y,
          batch_size=64,
          epochs=epoch_num,
          callbacks=[LambdaCallback(on_epoch_end=callback_ep)])

print("Save LSTM trained model")
model.save("./LSTM_TG.model")
#model = load_model("./LSTM_TG.model")

# this is used when you want to 
# assign a noun word as the 
# beginning of the poem for checking the generating system
"""
NN_words = set()
def noun_words_extract(sentences):
  for s in sentences:
    if type(s)!=float:
      s_tokens_C = word_tokenize(s)
      s_pos_tag = pos_tag(s_tokens_C)
      for i in s_pos_tag:
        if 'NN' in i[-1]:
          NN_words.add(i[0])
noun_words_extract(sentences_x)
noun_words_extract(sentences_y)
noun_words_extract(sentences_y_p)
"""

print("Generating haiku-like poem...\n")
random_start = random.choice(tuple(vocab))
haikulike = poem_generator(random_start)
print(haikulike[0].capitalize())
print(haikulike[1].capitalize())
print(haikulike[2].capitalize())


