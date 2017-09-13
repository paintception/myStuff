from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tensorboard.plugins import projector

import collections
import math
import os
import random
import zipfile
import time
import re

import numpy as np
from six.moves import urllib
from six.moves import xrange  
import tensorflow as tf

from sklearn.manifold import TSNE

LOG_DIR = ""  # For Tensorboard logs

common_words = 15
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label
valid_window = 100    # Only pick dev samples in the head of the distribution.
num_sampled = 50      # Number of negative examples to sample.
num_steps = 1001
lr = 0.01

potential_centroids = []  # Feed the list with the words you want to perfrom 
                          # the Word2Vec analysis on 

valid_size = len(potential_centroids)

def loadCorpus():
  return np.load('')  # Load the corpus 

def checkIfMail(email):
  match = re.match(r'[\w\.-]+@[\w\.-]+(\.[\w]+)+', email) # Useful Regex
  if match == None:
    return True

def checkIfOrderNumber(order):
  match = re.match(r'^[a-zA-Z]{3}\d{8}?$', order) # Useful Regex
  if match == None:
    return True  

vocabulary = loadCorpus()
vocabulary = [x.lower() for x in vocabulary]
vocabulary = [email for email in vocabulary if checkIfMail(email)]
vocabulary = [email for email in vocabulary if checkIfOrderNumber(email)]
vocabulary = set(vocabulary)
vocabulary_size = len(vocabulary)

def build_dataset(words, n_words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0 
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)


myvalid_examples = []

for i in count:
  for potential_centroid in potential_centroids:
    if i[0] == potential_centroid:
      idx = count.index(i)
      myvalid_examples.append(idx)

valid_examples = np.asarray(myvalid_examples)      

del vocabulary  

print('Most common words (+UNK)', count[:common_words])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
      for word in data[:span]:
        buffer.append(word)
    else:
      buffer.append(data[data_index])
      data_index += 1
  
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

graph = tf.Graph()

with graph.as_default():

  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  with tf.device('/cpu:0'):
    embedding_var = tf.Variable([vocabulary_size, embedding_size],'word_embedding')
    #config = projector.ProjectorConfig()
    #embedding = config.embeddings.add()
    #embedding.tensor_name = embedding_var.name

    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

  summary_writer = tf.summary.FileWriter(LOG_DIR)

  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  init = tf.global_variables_initializer()

with tf.Session(graph=graph) as session:
  init.run()
  print('Initialized')

  saver = tf.train.Saver()
  saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), 1)
  
  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      print('-------------------------------------------------------------')
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
        #saver.save(session, LOG_DIR+"blog.ckpt", step)
        summary_writer = tf.summary.FileWriter(LOG_DIR)

  final_embeddings = normalized_embeddings.eval()
  
tsne = TSNE(perplexity=30, n_components=4, init='pca', n_iter=1500, method='exact')
plot_only = vocabulary_size
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in xrange(plot_only)]

np.save('Output.npy', low_dim_embs) # Save word embeddings for further visualization
np.save('Labels.npy', labels)