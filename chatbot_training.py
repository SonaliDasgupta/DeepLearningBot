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
  
    
#Padding sequences with <PAD> token to make question and answer of same length#
def applyPadding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']]*(max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
   
    
#splitting the data into batches of questions and answers
import numpy as np
def splitIntoBatches(questions, answers, batch_size):
   
    numBatches = len(questions)//batch_size
    
    for batch_num in range(numBatches):
        current = batch_num * batch_size
        temp_batch_questions = [questions[current : (current + batch_size)]]
        temp_batch_answers = [answers[current : (current + batch_size)]]
        padded_batch_questions = np.array(applyPadding(temp_batch_questions, questionswords2int))
        padded_batch_answers = np.array(applyPadding(temp_batch_answers, answerswords2int))
        yield padded_batch_questions, padded_batch_answers
        
    
#splitting data into training and validation sets for both questions and answers
training_validation_split = int(len(sorted_cleaned_questions) * 0.15)
training_questions = sorted_cleaned_questions[training_validation_split: ]
training_answers = sorted_cleaned_answers[training_validation_split: ]
validation_questions = sorted_cleaned_questions[:training_validation_split]
validation_answers = sorted_cleaned_answers[:training_validation_split]


    #start training#
import time

batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_errors = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs+1):
    for batch_index , (batch_padded_questions, batch_padded_answers) in enumerate(splitIntoBatches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_grad_clipping, loss_error], {inputs: batch_padded_questions,
                                                                                           targets: batch_padded_answers,
                                                                                           lr: learning_rate,
                                                                                           sequence_length: batch_padded_answers.shape[1],
                                                                                           keep_prob: keep_probability } 
                                                                                                   
                                                   )
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_training_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch : {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training time on 100 batches: {:d} seconds'.format(epoch,
                                                                                                                                        epochs,
                                                                                                                                        batch_index,
                                                                                                                                        len(training_questions)/batch_size,
                                                                                                                                        total_training_loss_error/batch_index_check_training_loss,
                                                                                                                                       int(batch_training_time * batch_index_check_training_loss)
                                                                                                                                        ))
            total_training_loss_error = 0
        
        if batch_size % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation , (batch_padded_questions, batch_padded_answers) in enumerate(splitIntoBatches(validation_questions, validation_answers, batch_size)):
        
                batch_validation_loss_error = session.run(loss_error, {inputs: batch_padded_questions,
                                                                       targets: batch_padded_answers,
                                                                       lr: learning_rate,
                                                                       sequence_length: batch_padded_answers.shape[1],
                                                                       keep_prob: 1
                                                                      }
                                                                                                   
                                                   )
                
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_validation_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions)) // batch_size
        
            print('Validation loss error : {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error,
                                                                                                 batch_validation_time))
         
           #apply learning rate decay
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
           
           #early stopping
            list_validation_loss_errors.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_errors):
                print('I speak better now')
                early_stopping = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print('Sorry I do not speak better, Need to practice more')
                early_stopping += 1
                if early_stopping == early_stopping_stop:
                    break

    if early_stopping == early_stopping_stop:
        print('My apologies. I cannot speak any better. This is the best I can do')
        #early stopping
        break
print('Game Over')
               
       
            
        
   
    
        