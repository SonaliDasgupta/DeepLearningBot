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
#sequence_length =  tf.placeholder(tf.int32, shape = (batch_size), name = 'sequence_length')
sequence_length = tf.placeholder_with_default([25 for _ in range(batch_size)], shape=(batch_size), name = 'sequence_length')
sequence_length_foreach = tf.placeholder_with_default(25, None, name = 'sequence_length_for_each')
#Getting shape of input tensor
input_shape = tf.shape(inputs)
#print 'SHAPE OF TARGET: '+str(tf.shape(targets))

#Getting training and test predictions

training_predictions, test_predictions  = rnn.seq2seq_model(tf.reverse(inputs, [-1]), targets, keep_prob, batch_size,
                                                            sequence_length,
                                                            len(answerswords2int),
                                                            len(questionswords2int),
                                                            encoding_embedding_size,
                                                            decoding_embedding_size,
                                                            RNN_size,
                                                            num_layers,
                                                            questionswords2int)


#print 'SHAPE OF TRAINING PREDS: '+str(tf.shape(training_predictions))
#Setting up Loss Error , Optimizer and Gradient Clipping
with tf.name_scope('optimization'):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([batch_size, sequence_length_foreach]))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5.0 , 5.0), grad_variable)  for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_grad_clipping = optimizer.apply_gradients(clipped_gradients)