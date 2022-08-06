import json
import string
import random 
import os
import nltk
import numpy as np
nltk.download("punkt")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
import tensorflow as tf 
from keras import Sequential 
from keras.layers import Dense, Dropout

# load the intents json file
dirname = os.path.dirname(__file__)
pathToDataSet = os.path.join(dirname, "../data/dataset.json")
jsonfile = open(pathToDataSet)
data = json.load(jsonfile)
jsonfile.close()

# initializing lemmatizer to get stem of words_list
lemmatizer = WordNetLemmatizer()

# Each list to create
words_list = []
classes = []
doc_X = []
doc_y = []

# Loop through all the intents
# tokenize each pattern and append tokens to words_list, the patterns and
# the associated tag to their associated list
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # split pattern
        tokens = nltk.word_tokenize(pattern)
        # put all pattern tokens in the words_list list
        words_list.extend(tokens)
        # put pattern in doc_X
        doc_X.append(pattern)
        # put tag of the pattern doc_y
        doc_y.append(intent["tag"])
    
    # add the tag to the classes if it's not there already 
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
# lemmatize all the words_list in the vocab and convert them to lowercase
# if the words_list don't appear in punctuation
words_list = [lemmatizer.lemmatize(word.lower()) for word in words_list if word not in string.punctuation]
# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
words_list = sorted(set(words_list))
classes = sorted(set(classes))

# list for training data
training = []
# empty array length of classes array. default value of 0
out_empty = [0] * len(classes)
# creating the bag of words_list model
# enumerate gives each item index
for idx, doc in enumerate(doc_X):
    bow = []
    # lemmatize item and convert to lowercase
    text = lemmatizer.lemmatize(doc.lower())
    for word in words_list:
        # if pattern token in pattern, append 1. else append 0
        bow.append(1) if word in text else bow.append(0)
    # mark the index of class that the current pattern is associated
    # to
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    # add the one hot encoded BoW and associated classes to training 
    training.append([bow, output_row])
# shuffle the data and convert it to an array
random.shuffle(training)
training = np.array(training, dtype=object)
# split the features and target labels
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))


# defining some parameters
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200
# the deep learning model
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
# preventing deep learning models from overfitting to data
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))
# compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)

def clean_text(text): 
  # spilit the text into words_list
  tokens = nltk.word_tokenize(text)
  # remove all the punctuation
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  # return the clean text
  return tokens

def bag_of_words_list(text, vocab): 
  # clean the text
  tokens = clean_text(text)
  # create an empty bag of words_list
  bow = [0] * len(vocab)
  # loop through the tokens and add 1 to the bag of words_list if the token is in the vocab
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  # return the bag of words_list
  return np.array(bow)

def pred_class(text, labels): 
  # create the bag of words_list
  bow = bag_of_words_list(text, words_list)
  # predict the class
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  # return the class
  return return_list

def get_response(intents_list, intents_json): 
  # get the tag from the list of intents
  tag = intents_list[0]
  # get a list of intents associated with the tag
  list_of_intents = intents_json["intents"]
  for i in list_of_intents: 
    if i["tag"] == tag:
      # get a random response with the tag from the intent's list of responses
      result = random.choice(i["responses"])
      break
  # return the response
  return result
