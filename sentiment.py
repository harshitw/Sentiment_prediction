import tensorflow as tf
import numpy as np

with open('../sentiment-network/reviews.txt', 'r') as f:
    reviews = f.read()
with open('../sentiment-network/reviews.txt', 'r') as f:
    labels = f.read()

# reviews[:2000]

# DATA PREPROCESSING
# we will be using embedding layers insted of one hot vector representation
# because we have tons of thousand words in our datasets
from strings import punctuation
all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')
all_text = ' '.join(reviews)
words = all_text.split()

# all_text[:2000]
# words[:100]

from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key = counts.get, reverse = True)

# a dictionay that maps vocab words to integers
vocab_to_int = {word: i for i, word in enumerate(vocab)}
# converting the reviews to integers, same shape as review list but with integers
reviews_ints = []
for review in reviews:
    review_ints.append([vocab_to_int[word] for word in review.split()])

# encoding the labels
labels = labels.split('\n')
labels = np.array([1 if review == 'positive' else 0 for review in labels])

review_lens = Counter([len(x) for x in reviews_ints])
# num(review_lens == 0) = 1
# max(review_lens) == 256
# we'll keep the length of reviews to 200, we'll pad other reviews with zero that are less than 200
non_zero_reviews = [i for i, review in enumerate(review_ints) if len(review) != 0]
# len(non_zero_reviews)
reviews_ints = [reviews_ints[i] for i in non_zero_reviews]
labels = [labels[i] for i in non_zero_reviews]

seq_len = 200
features = np.zeros((len(reviews_ints), seq_len), dtype = int)
for i, review in enumerate(reviews_ints):
    features[i, -len(review):] =  np.array(review)[:seq_len]

# training and testing
split_frac = 0.8
idx = int(len(reviews_ints)*split_frac)
train_x, valid_x = reviews_ints[:idx], reviews_ints[idx:]
train_y, valid_y = features[:idx], features[idx:]

idx1 = int(len(valid_x)*0.5)
val_x, test_x = valid_x[:idx1], valid_x[idx1:]
val_y. test_y = valid_y[:idx1], valid_y[idx1:]

# defining the lstm cell size i.e. number of units in hidden layer in the lstm cells
lstm_size = 256
lstm_layers = 1
batch_size = 500
learning_rate = 0.001


n_words = len(vocab_to_int) + 1
graph = tf.Graph()
# adding the nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name = 'inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name = 'labels')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

# embeddings
# creating a embedding look up matrix to get the embedded vectors to pass in the LSTM cell
embed_size = 300
with graph.as_default():
    # uniform initialization with values between -1 to 1
    embedding = tf.Variable(tf.random_uniform(n_words, embed_size), -1, 1)
    embed = tf.nn.embedding_lookup(embedding, inputs_)

# LSTM cell
with graph.as_default():
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop]*lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

# RNN forward pass
outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state = initial_state)

# output
with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# validation accuracy
with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

....................................
