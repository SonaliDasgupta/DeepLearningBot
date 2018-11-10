# -*- coding: utf-8 -*-

import os
os.chdir('/home/admin1/DeepLearningBot')
import chatbot_NLP_preprocessing
import tensorflow as tf
        # BUILDING SEQ2SEQ MODEL #
  
class Seq2SeqModel:  
      
    #Creating placeholders for inputs and targets
    def model_inputs(self):
        inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
        targets = tf.placeholder(tf.int32, [None, None], name = 'target')
        lr = tf.placeholder(tf.float32, name = 'learning_rate')
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        return inputs, targets, lr, keep_prob

    #preprocess targets
    def preprocess_targets(self, targets, word2int, batch_size):
        left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
        right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
        preprocessed_targets = tf.concat([left_side, right_side], axis = 1)
        return preprocessed_targets
    
          
    #creating architecture of Seq2Seq Model
    
    #Encoder RNN layer
    def encoderRNN(self, rnn_inputs, rnn_size, num_layers, keep_prob, seq_length):
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        print lstm.state_size
        encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)  
        
        #add bidirectional later
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell = encoder_cell,
                                                                      
                                                               inputs = rnn_inputs,
                                                                         sequence_length = seq_length,                 
                                                                         dtype = tf.float32)
        print "ENCODER OUTPUT:"+ str(encoder_outputs)
        print "ENCODER STATE:"+ str(encoder_state)
        #return encoder_outputs
        return encoder_outputs, encoder_state


    #Decoder RNN Layer

    #Decoding the training set
    import numpy as np
    def decodeTrainingSet(self, encoder_outputs, encoder_state, decoder_cell, decoder_embedded_input, sequence_length,
                          decoding_scope, output_function, keep_prob, batch_size, rnn_size, num_layers):
      projection_layer = tf.layers.Dense(units = len(answersint2words), use_bias=True)
      num_units = rnn_size * num_layers
     # attention_states = tf.transpose(encoder_outputs, [batch_size, 0, rnn_size])

      helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedded_input,
                                                 sequence_length
                                                                 
                                                 )
     
      attention_mech = tf.contrib.seq2seq.BahdanauAttention(num_units = num_units, 
                                                    
                                                     memory = encoder_outputs,
                                                     memory_sequence_length = sequence_length)
      
      attn_cell = tf.contrib.seq2seq.AttentionWrapper(
        cell = decoder_cell,# Instance of RNNCell
        attention_mechanism = attention_mech, # Instance of AttentionMechanism
        attention_layer_size = num_units/2, # Int, depth of attention (output) tensor
        name="attention_wrapper")
      out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, len(answersint2words), reuse = tf.AUTO_REUSE
                )
     
      #change to BeamSearchDecoder
      decoder = tf.contrib.seq2seq.BasicDecoder(
              cell = out_cell,
              helper = helper, # A Helper instance 
              initial_state = out_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state),
              output_layer = projection_layer
             )
      
      
      # Perform dynamic decoding with decoder object
      outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
      rnn_out, sample_id = outputs
      rnn_out = tf.to_float(rnn_out)
      decoder_output_dropout = tf.nn.dropout(rnn_out, keep_prob)
     # print 'OUTPUT SHAPE: '+str(tf.shape(outputs))
    #  outputs = [tf.to_float(output) for output in outputs]
   #   print 'OUTPUT SHAPE: '+str(tf.shape(outputs))
      #Apply Dropout later
  #    decoder_output_dropout = tf.nn.dropout(outputs, keep_prob)
      return decoder_output_dropout
   #   return output_function(decoder_output_dropout)


    #Decoding test/Validation Set
    def decodeTestValidationSet(self, encoder_outputs, encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words,
                                decoding_scope, output_function, batch_size):
        attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
      #  attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units = decoder_cell.output_size, attention_states, memory_sequence_length = maximum_length)
        attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.AttentionWrapper(attention_states, num_units = decoder_cell.output_size, batch_size = batch_size)
        test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(
                output_function, encoder_state[0], attention_keys, attention_values, attention_score_function,
                attention_construct_function, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, name = "attn_dec_inf")
        test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                              test_decoder_function,
                                                                                                              decoding_scope)
        return test_predictions

    #Decoder RNN Layer
    def decoderRNN(self, decoder_embedded_input, decoder_embeddings_matrix, encoder_outputs, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
        with tf.variable_scope('decoding') as decoding_scope:
            lstm1 = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            lstm_dropout1 = [tf.contrib.rnn.DropoutWrapper(lstm1, input_keep_prob = keep_prob)]
           # layers = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
           # for _ in range(num_layers)]
            lstm2 = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            lstm_dropout2 = tf.contrib.rnn.DropoutWrapper(lstm2, input_keep_prob = keep_prob)
            lstm_dropout2 = [lstm_dropout2]*(num_layers-1)
            lstm_dropout = lstm_dropout1 + lstm_dropout2
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_dropout)
            
         #   decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
            weights = tf.truncated_normal_initializer(stddev = 0.1)
            biases = tf.zeros_initializer()
            output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                          num_words,
                                                                          None,
                                                                          scope = decoding_scope,
                                                                          weights_initializer = weights,
                                                                          biases_initializer = biases)
            training_predictions = self.decodeTrainingSet(encoder_outputs, encoder_state, decoder_cell,
                                                 decoder_embedded_input,
                                                 sequence_length,
                                                 decoding_scope,
                                                 output_function,
                                                 keep_prob,
                                                 batch_size,
                                                 rnn_size, num_layers)
            print 'TRAINING PREDS: '+str(training_predictions)
            decoding_scope.reuse_variables()
            test_predictions = self.decodeTestValidationSet(encoder_outputs, encoder_state,
                                                   decoder_cell,
                                                   decoder_embeddings_matrix,
                                                   word2int['<SOS>'],
                                                   word2int['<EOS>'],
                                                   sequence_length-1,
                                                   num_words,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
            return training_predictions, test_predictions
    

    #final brain model
    def seq2seq_model(self, inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
        encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                                  answers_num_words+1,
                                                                  encoder_embedding_size,
                                                                  initializer = tf.random_uniform_initializer(0,1))
        encoder_outputs, encoder_state = self.encoderRNN(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)

        preprocessed_targets = self.preprocess_targets(targets, questionswords2int, batch_size)
        decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1, decoder_embedding_size],0,1))
        decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
        training_predictions, test_predictions = self.decoderRNN(decoder_embedded_input, decoder_embeddings_matrix,
                                                        encoder_outputs, encoder_state,
                                                        questions_num_words,
                                                        sequence_length,
                                                        rnn_size,
                                                        num_layers,
                                                        questionswords2int,
                                                        keep_prob,
                                                        batch_size)
        #return training_predictions, test_predictions"""
    
    
    

    