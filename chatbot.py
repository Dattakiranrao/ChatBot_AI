import tensorflow as tf 
from tensorflow.python.framework import ops
import tflearn as tfl 
import json
import numpy as np 
import random
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open('intents.json', 'rb') as f:
    data = json.load(f)

try:
    with open('chatbot_data.pickle', 'rb') as f:
        all_words, labels, training_data, output_data = pickle.load(f)
except:
    all_words = []
    labels = []
    every_pattern = []
    associate_labels = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            words_in_pattern = nltk.word_tokenize(pattern)
            all_words.extend(words_in_pattern)
            every_pattern.append(words_in_pattern)
            associate_labels.append(intent['tag'])
            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    all_words = [stemmer.stem(w.lower()) for w in all_words if w not in '?']
    all_words = sorted(list(set(all_words))) 

    training_data = []
    output_data = []

    label_copy = [0 for _ in range(len(labels))]

    for index, word in enumerate(every_pattern):
        bag = []

        words_in_pattern = [stemmer.stem(w) for w in word]

        for w in all_words:
            if w in words_in_pattern:
                bag.append(1)
            else:
                bag.append(0)

        tag_decider = label_copy[:]
        tag_decider[labels.index(associate_labels[index])] = 1 
        training_data.append(bag)
        output_data.append(tag_decider)

    training_data = np.array(training_data)
    output_data = np.array(output_data)

    with open('chatbot_data.pickle', 'wb') as f:
        pickle.dump((all_words, labels, training_data, output_data), f)

ops.reset_default_graph()

net = tfl.input_data(shape=[None, len(training_data[0])])
net = tfl.fully_connected(net, 8) 
net = tfl.fully_connected(net, 8) 
net = tfl.fully_connected(net, len(output_data[0]), activation='softmax') 
net = tfl.regression(net)

model = tfl.DNN(net)

try:
    model.load('chatbot_model.pickle')
except:
    model.fit(training_data, output_data, n_epoch=10000, batch_size=8, show_metric=True)
    model.save('chatbot_model.pickle')

def bag_of_words(user_input_words, all_words):
    bag = [0 for _ in range(len(all_words))]

    user_input_words = nltk.word_tokenize(user_input_words)
    user_input_words = [stemmer.stem(w.lower()) for w in user_input_words]

    for word in user_input_words:
        for index, word_in_all_words in enumerate(all_words):
            if word == word_in_all_words:
                bag[index] = 1
    return np.array(bag)

def chat():
    while True:
        user_input = input("Question: ")
        if user_input.lower() == 'quit':
            break

        predection = model.predict([bag_of_words(user_input, all_words)])[0]
        index_result = np.argmax(predection)
        tag = labels[index_result]
        
        if predection[index_result] > 0.7:
            for tg in data['intents']:
                if tg['tag'] == tag :
                    response = tg['responses']
            print(random.choice(response))
        else:
            print("Please Ask A Different Question..")


if __name__ == "__main__":
    chat()


