# -*- coding: utf-8 -*-

#importing the libraries
import numpy as np
import tensorflow as tf
import re
import time



                ###DATA PREPROCESSING###
                
                
#Read the lines and conversations (list of lines)
lineBytes = open('/home/admin1/DeepLearningBot/movie_lines.txt').read()
lines = lineBytes.decode('utf-8','ignore').split('\n')
conversationBytes = open('/home/admin1/DeepLearningBot/movie_conversations.txt').read()
conversations = conversationBytes.decode('utf-8','ignore').split('\n')


#provide ID to each dialog and store in list
id2dialog = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) >= 5:
        id2dialog[_line[0]] = _line[4]
    
conversation_ids = []
for conversation in conversations:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversation_ids.append(_conversation.split(','))
    
    
"""store questions and answers, by iterating the list of lines, a line is a 
question with its next line as answer, in the same conversation"""    
questions = []
answers = []
for conversation in conversation_ids:
    for i in range(len(conversation)-1):
        questions.append(id2dialog[conversation[i]])
        answers.append(id2dialog[conversation[i+1]])
        
#clean the questions and answers text
def cleanText(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'re", " are", text) 
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'s", " us", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"'nt", " not", text)
    text = re.sub(r"[-,()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

cleaned_questions = []
for question in questions:
    cleaned_questions.append(cleanText(question))
    
cleaned_answers = []
for answer in answers:
    cleaned_answers.append(cleanText(answer))

#compute the frequencies of the words in the conversations
word2occurences = {}
sentences = cleaned_questions + cleaned_answers
for sentence in sentences:
    for word in sentence.split():
        if word not in word2occurences:
            word2occurences[word] = 1
        else:
            word2occurences[word] += 1
            

#discard words having less than 20 occurences so as to enable neural network (RNN) to train faster
threshold = 20
questionswords2int = {}
word_number = 0
for word, count in word2occurences.items():
    if count >= threshold:
        questionswords2int[word]= word_number
        word_number += 1 

word_number = 0
answerswords2int = {}
for word, count in word2occurences.items():
    if count >= threshold:
        answerswords2int[word]= word_number
        word_number += 1 
        

"""sum =0
for key, value in answerswords2int.items():
    for key1, value1 in questionswords2int.items():
        if key != key1 or value != value1:
            sum+=1"""
 
#add essential tokens to position dictionaries of words used in questions and answers           
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int)+1
    answerswords2int[token] = len(answerswords2int)+1
    
#form reverse dictionary for answers with index to word mapping for RNN decoding phase  
answersint2words = {w_i: w for w, w_i in answerswords2int.items()}

#add end of string character to each answer for RNN training
for i in range(len(cleaned_answers)):
    cleaned_answers[i]+= ' <EOS>'
 
    
#convert questions and answers to list of integers using the word indices in the dictionaries    
questions_to_int = []
for question in cleaned_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_to_int.append(ints)
    
answers_to_int = []
for answer in cleaned_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_to_int.append(ints)
    
    
#sort the questions and answers lists according to question length , for optimized training of RNN
sorted_cleaned_questions = []
sorted_cleaned_answers = []
for length in range(1, 25+1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_cleaned_questions.append(questions_to_int[i[0]])
            sorted_cleaned_answers.append(answers_to_int[i[0]])
            
"""for i in enumerate(sorted_cleaned_questions):
    if len(i[1]) == 2:
        print i[0]
        break"""
    

        

    

        
