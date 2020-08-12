#!/usr/bin/env python
# coding: utf-8

# Exercise 11 - Recurrent Neural Networks
# ========
# 
# A recurrent neural network (RNN) is a class of neural network that excels when your data can be treated as a sequence - such as text, music, speech recognition, connected handwriting, or data over a time period. 
# 
# RNN's can analyse or predict a word based on the previous words in a sentence - they allow a connection between previous information and current information.
# 
# This exercise looks at implementing a LSTM RNN to generate new characters after learning from a large sample of text. LSTMs are a special type of RNN which dramatically improves the model’s ability to connect previous data to current data where there is a long gap.
# 
# We will train an RNN model using a novel written by H. G. Wells - The Time Machine.

# Step 1
# ------
# 
# Let's start by loading our libraries and text file. This might take a few minutes.
# 
# #### Run the cell below to import the necessary libraries.

# In[1]:


get_ipython().run_cell_magic(u'capture', u'', u'# Run this!\nfrom keras.models import load_model\nfrom keras.models import Sequential\nfrom keras.layers import Dense, Activation, LSTM\nfrom keras.callbacks import LambdaCallback, ModelCheckpoint\nimport numpy as np\nimport random, sys, io, string')


# #### Replace the `&lt;addfilename&gt;` with `The Time Machine`&lt;/addfilename&gt;

# In[2]:


###
# REPLACE THE <addFileName> BELOW WITH The Time Machine
###
text = io.open('Data/The Time Machine.txt', encoding = 'UTF-8').read()
###

# Let's have a look at some of the text
print(text[0:198])

# This cuts out punctuation and make all the characters lower case
text = text.lower().translate(str.maketrans("", "", string.punctuation))

# Character index dictionary
charset = sorted(list(set(text)))
index_from_char = dict((c, i) for i, c in enumerate(charset))
char_from_index = dict((i, c) for i, c in enumerate(charset))

print('text length: %s characters' %len(text))
print('unique characters: %s' %len(charset))


# Expected output:  
# ```The Time Traveller (for so it will be convenient to speak of him) was expounding a recondite matter to us. His pale grey eyes shone and twinkled, and his usually pale face was flushed and animated.
# text length: 174201 characters
# unique characters: 39```
# 
# Step 2
# -----
# 
# Next we'll divide the text into sequences of 40 characters.
# 
# Then for each sequence we'll make a training set - the following character will be the correct output for the test set.
# 
# ### In the cell below replace:
# #### 1. `&lt;sequencelength&gt;` with `40`
# #### 2. `&lt;step&gt;` with `4`
# #### and then __run the code__. &lt;/step&gt;&lt;/sequencelength&gt;

# In[3]:


###
# REPLACE <sequenceLength> WITH 40 AND <step> WITH 4
###
sequence_length = 40
step = 4
###

sequences = []
target_chars = []
for i in range(0, len(text) - sequence_length, step):
    sequences.append([text[i: i + sequence_length]])
    target_chars.append(text[i + sequence_length])
print('number of training sequences:', len(sequences))


# Expected output:
# `number of training sequences: 43541`
# 
# #### Replace `&lt;addsequences&gt;` with `sequences` and run the code.&lt;/addsequences&gt;

# In[4]:


# One-hot vectorise

X = np.zeros((len(sequences), sequence_length, len(charset)), dtype=np.bool)
y = np.zeros((len(sequences), len(charset)), dtype=np.bool)

###
# REPLACE THE <addSequences> BELOW WITH sequences
###
for n, sequence in enumerate(sequences):
###
    for m, character in enumerate(list(sequence[0])):
        X[n, m, index_from_char[character]] = 1
    y[n, index_from_char[target_chars[n]]] = 1


# Step 3
# ------
# 
# Let's build our model, using a single LSTM layer of 128 units. We'll keep the model simple for now, so that training does not take too long.
# 
# ### In the cell below replace:
# #### 1. `&lt;addlstm&gt;` with `LSTM`
# #### 2. `&lt;addlayersize&gt;` with `128`
# #### 3. `&lt;addsoftmaxfunction&gt;` with `'softmax`
# #### and then __run the code__.&lt;/addsoftmaxfunction&gt;&lt;/addlayersize&gt;&lt;/addlstm&gt;

# In[6]:


model = Sequential()

###
# REPLACE THE <addLSTM> BELOW WITH LSTM (use uppercase) AND <addLayerSize> WITH 128
###
model.add(LSTM(128, input_shape = (X.shape[1], X.shape[2])))
###

###
# REPLACE THE <addSoftmaxFunction> with 'softmax' (INCLUDING THE QUOTES)
###
model.add(Dense(y.shape[1], activation = 'softmax'))
###

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam')


# The code below generates text at the end of an epoch (one training cycle). This allows us to see how the model is performing as it trains. If you're making a large neural network with a long training time it's useful to check in on the model as see if the text generating is legible as it trains, as overtraining may occur and the output of the model turn to nonsense.
# 
# The code below will also save a model if it is the best performing model, so we can use it later.
# 
# #### Run the code below, but don't change it

# In[7]:


# Run this, but do not edit.
# It helps generate the text and save the model epochs.

# Generate new text
def on_epoch_end(epoch, _):
    diversity = 0.5
    print('\n### Generating text with diversity %0.2f' %(diversity))

    start = random.randint(0, len(text) - sequence_length - 1)
    seed = text[start: start + sequence_length]
    print('### Generating with seed: "%s"' %seed[:40])

    output = seed[:40].lower().translate(str.maketrans("", "", string.punctuation))
    print(output, end = '')

    for i in range(500):
        x_pred = np.zeros((1, sequence_length, len(charset)))
        for t, char in enumerate(output):
            x_pred[0, t, index_from_char[char]] = 1.

        predictions = model.predict(x_pred, verbose=0)[0]
        exp_preds = np.exp(np.log(np.asarray(predictions).astype('float64')) / diversity)
        next_index = np.argmax(np.random.multinomial(1, exp_preds / np.sum(exp_preds), 1))
        next_char = char_from_index[next_index]

        output = output[1:] + next_char

        print(next_char, end = '')
    print()
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# Save the model
checkpoint = ModelCheckpoint('Models/model-epoch-{epoch:02d}.hdf5', 
                             monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')


# The code below will start the model to train. This may take a long time. Feel free to stop the training with the `square stop button` to the right of the `Run button` in the toolbar.
# 
# Later in the exercise, we will load a pretrained model.
# 
# ### In the cell below replace:
# #### 1. `&lt;addprintcallback&gt;` with `print_callback`
# #### 2. `&lt;addcheckpoint&gt;` with `checkpoint`
# #### and then __run the code__.&lt;/addcheckpoint&gt;&lt;/addprintcallback&gt;

# In[8]:


###
# REPLACE <addPrintCallback> WITH print_callback AND <addCheckpoint> WITH checkpoint
###
model.fit(X, y, batch_size = 128, epochs = 3, callbacks = [print_callback, checkpoint])
###


# The output won't appear to be very good. But then, this dataset is small, and we have trained it only for a short time using a rather small RNN. How might it look if we upscaled things?
# 
# Step 5
# ------
# 
# We could improve our model by:
# * Having a larger training set.
# * Increasing the number of LSTM units.
# * Training it for longer
# * Experimenting with difference activation functions, optimization functions etc
# 
# Training this would still take far too long on most computers to see good results - so we've trained a model already for you.
# 
# This model uses a different dataset - a few of the King Arthur tales pasted together. The model used:
# * sequences of 50 characters
# * Two LSTM layers (512 units each)
# * A dropout of 0.5 after each LSTM layer
# * Only 30 epochs (we'd recomend 100-200)
# 
# Let's try importing this model that has already been trained.
# 
# #### Replace `&lt;addloadmodel&gt;` with `load_model` and run the code.&lt;/addloadmodel&gt;

# In[9]:


from keras.models import load_model
print("loading model... ", end = '')

###
# REPLACE <addLoadModel> BELOW WITH load_model
###
model = load_model('Models/arthur-model-epoch-30.hdf5')
###
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam')
###

print("model loaded")


# Step 6
# -------
# 
# Now let's use this model to generate some new text!
# 
# #### Replace `&lt;addfilepath&gt;` with `'Data/Arthur tales.txt'`&lt;/addfilepath&gt;

# In[10]:


###
# REPLACE <addFilePath> BELOW WITH 'Data/Arthur tales.txt' (INCLUDING THE QUOTATION MARKS)
###
text = io.open('Data/Arthur tales.txt', encoding='UTF-8').read()
###

# Cut out punctuation and make lower case
text = text.lower().translate(str.maketrans("", "", string.punctuation))

# Character index dictionary
charset = sorted(list(set(text)))
index_from_char = dict((c, i) for i, c in enumerate(charset))
char_from_index = dict((i, c) for i, c in enumerate(charset))

print('text length: %s characters' %len(text))
print('unique characters: %s' %len(charset))


# ### In the cell below replace:
# #### 1. `&lt;sequencelength&gt;` with `50`
# #### 2. `&lt;writesentence&gt;` with a sentence of your own, at least 50 characters long.
# #### 3. `&lt;numcharstogenerate&gt;` with the number of characters you want to generate (choose a large number, like 1500)
# #### and then __run the code__.&lt;/numcharstogenerate&gt;&lt;/writesentence&gt;&lt;/sequencelength&gt;

# In[16]:


# Generate text

diversity = 0.5
print('\n### Generating text with diversity %0.2f' %(diversity))

###
# REPLACE <sequenceLength> BELOW WITH 50
###
sequence_length = 50
###

# Next we'll make a starting point for our text generator

###
# REPLACE <writeSentence> WITH A SENTENCE OF AT LEAST 50 CHARACTERS
###
seed = "Mi corazon es como un dios sin lengua, Mudo se esta a la espera del milagro,He amado mucho, todo amor fue magro,Que todo amor lo conoci con mengua.He amado hasta llorar, hasta morirme.Ame hasta odiar, ame hasta la locura,Pero yo espero algun amor naturaCapaz de renovarme y redimirme.Amor que fructifique mi desierto. Y me haga brotar ramas sensitivas,Soy una selva de raices vivas,Solo el follaje suele estarse muerto.¿En donde esta quien mi deseo alienta?"
###

seed = seed.lower().translate(str.maketrans("", "", string.punctuation))

###
# OR, ALTERNATIVELY, UNCOMMENT THE FOLLOWING TWO LINES AND GRAB A RANDOM STRING FROM THE TEXT FILE
###

#start = random.randint(0, len(text) - sequence_length - 1)
#seed = text[start: start + sequence_length]

###

print('### Generating with seed: "%s"' %seed[:40])

output = seed[:sequence_length].lower().translate(str.maketrans("", "", string.punctuation))
print(output, end = '')

###
# REPLACE THE <numCharsToGenerate> BELOW WITH THE NUMBER OF CHARACTERS WE WISH TO GENERATE, e.g. 1500
###
for i in range(1500):
###
    x_pred = np.zeros((1, sequence_length, len(charset)))
    for t, char in enumerate(output):
        x_pred[0, t, index_from_char[char]] = 1.

    predictions = model.predict(x_pred, verbose=0)[0]
    exp_preds = np.exp(np.log(np.asarray(predictions).astype('float64')) / diversity)
    next_index = np.argmax(np.random.multinomial(1, exp_preds / np.sum(exp_preds), 1))
    next_char = char_from_index[next_index]

    output = output[1:] + next_char

    print(next_char, end = '')
print()


# How does it look? Does it seem intelligible?
# 
# Conclusion
# --------
# 
# We have trained an RNN that learns to predict characters based on a text sequence. We have trained a lightweight model from scratch, as well as imported a pre-trained model and generated new text from that.
