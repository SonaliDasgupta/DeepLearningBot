#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import chatbot_RNN

rnn = Seq2SeqModel()

    #Model TRAINING#

#Setting hyperparamters
epochs = 100
batch_size = 64
RNN_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

#Defining a session
import tensorflow as tf
tf.reset_default_graph()
session = tf.InteractiveSession()

#Loading model inputs
inputs, targets, lr, keep_prob = rnn.model_inputs()

#Setting sequence length
sequence_length =  tf.placeholder(tf.int32, shape = (batch_size), name = 'sequence_length')

#Getting shape of input tensor
input_shape = tf.shape(inputs)

#Getting training and test predictions

#training_predictions, test_predictions  = 
rnn.seq2seq_model(tf.reverse(inputs, [-1]), targets, keep_prob, batch_size,
                  sequence_length,
                  len(answerswords2int),
                  len(questionswords2int),
                  encoding_embedding_size,
                  decoding_embedding_size,
                  RNN_size,
                  num_layers,
                  questionswords2int)