#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import de[pendencies
import numpy


# In[24]:


import sys
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint


# In[3]:


#load data
file = open('Frankenstein.txt').read()


# In[4]:


#tokenisation and #standardisation
def tokenize_words(input):
    input = input.lower()
    tokenizer= RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)
    filtered = filter(lambda token : token not in stopwords.words('english'), tokens)
    return " ".join(filtered)

processed_inputs = tokenize_words(file)


# In[5]:


#char to numbers
chars = sorted(list(set(processed_inputs)))
char_to_num= { ch:i for i,ch in enumerate(sorted(chars)) }


# In[6]:


#check if words to char and chars to num has worked
input_len = len(processed_inputs)
vocab_len = len(chars)
print(input_len , vocab_len)


# In[7]:


#seg length
seq_len  = 100
x_data= []
y_data = []


# In[8]:


#loop through the sequence
for i in range(0 , input_len - seq_len,1):
    in_seq = processed_inputs[i:i + seq_len]
    out_seq = processed_inputs[i + seq_len]
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])
    
n_patterns = len(x_data)
print('total patterns' , n_patterns)


# In[9]:


#convert input sequence to np array
X= numpy.reshape(x_data, (n_patterns, seq_len, 1))
X= X/float(vocab_len)


# In[10]:


#one hot encoding
y = np_utils.to_categorical(y_data)


# In[15]:


#creating the model
#dropout to prevent overfitting
model =Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


# In[16]:


#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[26]:


#saving weights
filepath= "model_weights_saved.hdf5"
checkpoint= ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only= True, mode= 'min' )
desired_callbacks = [checkpoint]


# In[38]:


#fit model and let it train
model.fit(X,y, epochs=50, batch_size=256, callbacks=desired_callbacks)


# In[39]:


#recompile model with saved weights
filename= "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[40]:


#output of the model back into characters
num_to_char = dict((i,c) for i,c in enumerate(chars))


# In[41]:


#random seed to help generate
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print('Random Seed:')
print("\"",''.join([num_to_char[value] for value in pattern]),"\"")


# In[42]:


#generate the text
for i in range(1000):
    x= numpy.reshape(pattern, (1,len(pattern), 1))
    x= x/float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result= num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]


# In[ ]:




