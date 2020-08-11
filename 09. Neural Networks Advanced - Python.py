#!/usr/bin/env python
# coding: utf-8

# Exercise 9 - Advanced Neural Networks
# ==========
# 
# There are many factors that influence how well a neural network might perform. AI practitioners tend to play around with the structure of the hidden layers, the activation functions used, and the optimisation function.
# 
# In this exercise we will look at how changing these parameters impacts the accuracy performance of our network.

# Step 1
# ------
# 
# In this exercise we will use the same dog dataset as in exercise 8, building on what we learnt before and trying different parameters for a network to try and improve performance.
# 
# Let's start by opening up our data set and setting up our train and test sets.
# 
# #### __Run the code__ below.

# In[15]:


# Run this!

# Here we set a randomisation seed for replicatability.
import os
os.environ['PYTHONHASHSEED'] = '0'
seed = 6
import random as rn
rn.seed(seed)
import numpy as np
np.random.seed(seed)

import warnings
warnings.filterwarnings("ignore")

from keras import backend as K
import keras

print('keras using %s backend'%keras.backend.backend())
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# Sets up the graphing configuration
import matplotlib.pyplot as graph
get_ipython().magic(u'matplotlib inline')
graph.rcParams['figure.figsize'] = (15,5)
graph.rcParams["font.family"] = 'DejaVu Sans'
graph.rcParams["font.size"] = '12'
graph.rcParams['image.cmap'] = 'rainbow'


# In[16]:


# Run this too!
# This gets our data ready

# Load the data
dataset = pd.read_csv('Data/dog_data.csv')

# Separate out the features
features = dataset.drop(['breed'], axis = 1)

# Sets the target one-hot vectors
target = OneHotEncoder(sparse = False).fit_transform(np.transpose([dataset['breed']]))

# Take the first 4/5 of the data and assign it to training
train_X = features.values[:160]
train_Y = target[:160]

# Take the last 1/5 of the data and assign it to testing
test_X = features.values[160:]
test_Y = target[160:]


# Step 2
# ------
# 
# The box below contains methods to help us quickly change the structure. Don't edit them - just run the box.
# 
# The __train_network__ method allows us to change:
# * the number of layers
# * the activation functions the layers use
# * the optimizer of the model
# * the number of training cycles for the model (__epochs__)
# 
# The plot_acc and bar_acc just plot our models so we can easily see how well they do.
# 
# Don't worry about the code - it is simply to make the next steps easier.
# 
# #### __Run the code__ below.

# In[27]:


# Run this!
# Below are a few helper methods. Do not edit these.

def train_network(structure, activation, optimizer, epochs):
    
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(seed)
    np.random.seed(seed)
    
    # This initialises the model
    model = keras.models.Sequential()
    
    # This is our input + the first hidden layer 1
    model.add(keras.layers.Dense(units = structure[1], input_dim = structure[0], activation = activation)) 
    
    # Hidden layer 2, if not ignored (of size 0)
    if structure[2] > 0:
        model.add(keras.layers.Dense(units = structure[2], activation = activation))
        
    # Output layer
    model.add(keras.layers.Dense(units=structure[-1], activation = "softmax"))
    
    # Compiles the model with parameters
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
    
    # This tells the us training has started, so we know that it's actually running
    print('training... ', end = '')

    # This trains the network
    training_stats = model.fit(train_X, train_Y, batch_size = 1, epochs = epochs, verbose = 0, shuffle = False)
    
    # Results!
    print('train_acc: %0.3f, test_acc: %0.3f' %(training_stats.history['acc'][-1], 
                                                model.evaluate(test_X, test_Y, verbose = 0)[1]))
    
    # This returns the results and the model for use outside the function
    return training_stats, model

# Plots our evaluations in a line graph to see how they compare
def plot_acc(train_acc, test_acc, title):
    # Plots the training and testing accuracy lines
    training_accuracy, = graph.plot(train_acc, label = 'Training Accuracy')
    testing_accuracy, = graph.plot(test_acc, label = 'Testing Accuracy')
    graph.legend(handles = [training_accuracy, testing_accuracy])
    
    # Plots guide lines along y = 0 and y = 1 to help visualise
    xp = np.linspace(0, train_acc.shape[0] - 1, 10 * train_acc.shape[0])
    graph.plot(xp, np.full(xp.shape, 1), c = 'k', linestyle = ':', alpha = 0.5)
    graph.plot(xp, np.full(xp.shape, 0), c = 'k', linestyle = ':', alpha = 0.5)
    
    graph.xticks(range(0, train_acc.shape[0]), range(1, train_acc.shape[0] + 1))
    graph.ylim(0,1)
    graph.title(title)
    
    graph.show()

# Plots our evaluations in a bar chart to see how they compare
def bar_acc(train_acc, test_acc, title, xticks):
    index = range(1, train_acc.shape[0] + 1)
    
    # Plots the training and testing accuracy bars
    training_accuracy = graph.bar(index, train_acc, 0.4, align = 'center')
    testing_accuracy = graph.bar(index, test_acc, 0.4, align = 'edge')
    graph.legend((training_accuracy[0], testing_accuracy[0]), ('Training Accuracy', 'Testing Accuracy'))
    
    graph.xticks(index, xticks)
    graph.title(title)
    
    graph.show()


# Step 3
# ------
# 
# Let's first look at how different layer sizes impact performance.
# 
# Let's look at a network with just one hidden layer. We'll see how it performs with 1 to 10 nodes.
# 
# ### In the cell below replace:
# #### 1. `&lt;addhidden1&gt;` with `hidden1`
# #### 2. `&lt;addtrainacc&gt;` with `train_acc`
# #### 3. `&lt;addtestacc&gt;` with `test_acc`
# #### and then __run the code__.&lt;/addtestacc&gt;&lt;/addtrainacc&gt;&lt;/addhidden1&gt;

# In[29]:


# Initialises empty arrays into which to append new values.
train_acc = np.empty((0))
test_acc = np.empty((0))

for hidden1 in range (1,11):
    print('Evaluating model with %i hidden neurons... ' % hidden1, end = '')

###
# REPLACE <addHidden1> BELOW WITH hidden1
###
    training_stats, model = train_network(structure = [3, hidden1, hidden1, 3],activation = 'relu', optimizer = 'RMSprop', epochs = 12)
###
    
    train_acc = np.append(train_acc, training_stats.history['acc'][-1])
    test_acc = np.append(test_acc, model.evaluate(test_X, test_Y, verbose = 0)[1])

###
# REPLACE <addTrainAcc> WITH train_acc AND <addTestAcc> WITH test_acc
###
plot_acc(train_acc, test_acc, 'hidden layer size performance comparison')
###


# So, experimenting with different sizes of hidden layers can dramatically improve your results.
# 
# Step 4
# ------
# 
# Now we'll look at how different activation functions impact the performance.
# 
# There's lots we will try, just remember it is common to try both `relu` and `tanh` first.
# 
# ### In the cell below replace:
# #### 1. `&lt;addactivation&gt;` with `activation`
# #### 2. `&lt;addactivationfunctions&gt;` with `activation_functions`
# #### and then __run the code__.&lt;/addactivationfunctions&gt;&lt;/addactivation&gt;

# In[32]:


train_acc = np.empty((0))
test_acc = np.empty((0))

# Makes a list of the activation functions we wish to compare
activation_functions = ['elu', 'selu', 'relu', 'tanh', 'sigmoid', 
                        'hard_sigmoid', 'softplus', 'softsign', 'linear']

for activation in activation_functions:
    print('Evaluating model with %s hidden layer activation function... ' %activation, end = '')

###
# REPLACE <addActivation> WITH activation
###
    training_stats, model = train_network(structure = [3, 4, 2, 3],
                                          activation = activation, optimizer = 'RMSprop', epochs = 12)
###
    
    train_acc = np.append(train_acc, training_stats.history['acc'][-1])
    test_acc = np.append(test_acc, model.evaluate(test_X, test_Y, verbose=0)[1])
    
###
# REPLACE THE <addActivationFunctions> BELOW WITH activation_functions
###
bar_acc(train_acc, test_acc, 'activation function performance comparison using (4,2) hidden layer', activation_functions)
###


# There's quite a lot of variance there. It's always good to quickly test different activation functions first.
# 
# Next, lets try changing the shape of the hidden layers.
# 
# #### Replace `&lt;updatehere&gt;`'s with `3` and run the code.&lt;/updatehere&gt;

# In[34]:


train_acc = np.empty((0))
test_acc = np.empty((0))

activation_functions = ['elu', 'selu', 'relu', 'tanh', 'sigmoid',
                        'hard_sigmoid', 'softplus', 'softsign', 'linear']

for activation in activation_functions:
    print('Evaluating model with %s hidden layer activation function... ' %activation, end='')
    

# The value you choose for <updateHere> below will change the size of the hidden layers. Lets try changing them both to 3 for now
# (but you can have a play around with different numbers if you want)
###
# REPLACE THE <updateHere>'s BELOW WITH 3
###
    training_stats, model = train_network(structure = [3, 3, 3, 3], 
                                          activation = activation, optimizer = 'RMSprop', epochs = 12)
###
    
    train_acc = np.append(train_acc, training_stats.history['acc'][-1])
    test_acc = np.append(test_acc, model.evaluate(test_X, test_Y, verbose=0)[1])
    
bar_acc(train_acc, test_acc, 'activation function performance comparison using (3,3) hidden layer', activation_functions)


# Step 5
# -----
# 
# The __optimisation function__ is the last major parameter of the network architecture. It changes how the network is trained - so it can have a __very large impact on training time and end performance__.
# 
# Note: this step won't always provide the same results every time it is run. Optimizers such as SGD will give different results.
# 
# #### Replace `&lt;addoptimizer&gt;` with `optimizer` and run the code.&lt;/addoptimizer&gt;

# In[35]:


train_acc = np.empty((0))
test_acc = np.empty((0))

# This is a list of the optimisation functions for us to compare
optimization_functions = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta',
                          'Adam', 'Adamax', 'Nadam']

for optimizer in optimization_functions:
    print('Evaluating model with %s optimizer... ' %optimizer, end='')
    
    
# The <addOptimizer> below is where we specify the optimizer in the code    
###
# REPLACE THE <addOptimizer> BELOW WITH optimizer
###
    training_stats, model = train_network(structure = [3, 4, 2, 3],
                                          activation = 'relu', optimizer = optimizer, epochs = 12)
###

# This is recording our data for the plot
    train_acc = np.append(train_acc, training_stats.history['acc'][-1])
    test_acc = np.append(test_acc, model.evaluate(test_X, test_Y, verbose=0)[1])

# And now, the plot!    
bar_acc(train_acc, test_acc, 'optimizer performance comparison using (4,2) hidden layer', optimization_functions)


# Step 6
# -------
# 
# Let's try to combine what we've seen above and try to create a neural network that performs better than what we made in exercise 7, where we used the structure `[3,4,2,3]`, the activation function `relu`, and the optimiser `SGD` (Stochastic Gradient Descent).
# 
# ### In the cell below replace:
# #### 1. `&lt;layersize&gt;`'s with numbers of your choice (how many nodes the hidden layers will have)
# #### 2. `&lt;activationfunction&gt;` with one of the following: `'relu'`, `'softsign'`, `'tanh'`, `'elu'`, `'selu'`, `'softplus'`, `'linear'`
# #### 3. `&lt;optimiser&gt;` with one of the following: `'SGD'`, `'adam'`, `'RMSprop'`, `'Adagrad'`, `'Adadelta'`, `'Adamax'`, `'Nadam'`
# #### and then __run the code__.&lt;/optimiser&gt;&lt;/activationfunction&gt;&lt;/layersize&gt;

# In[39]:


###
# REPLACE THE <layerSize>'s' BELOW WITH PARAMETERS TO TEST A NEW NEURAL NETWORK e.g. 4 and 2
###
structure = [3, 5, 5, 3]
###

###
# REPLACE <activationFunction> WITH ONE OF THE FOLLOWING: 'relu', 'softsign', 'tanh', 'elu', 'selu', 'softplus', 'linear'
###
activation = 'tanh'
###

###
# REPLACE <optimiser> WITH ONE OF THE FOLLOWING: 'SGD', 'adam', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'
###
optimizer = 'Nadam'
###

training_stats, model = train_network(structure, activation, optimizer, epochs = 24)

# We can plot our training statistics to see how it developed over time
accuracy, = graph.plot(training_stats.history['acc'], label = 'Accuracy')
training_loss, = graph.plot(training_stats.history['loss'], label = 'Training Loss')
graph.legend(handles = [accuracy, training_loss])
loss = np.array(training_stats.history['loss'])
xp = np.linspace(0, loss.shape[0], 10 * loss.shape[0])
graph.plot(xp, np.full(xp.shape, 1), c = 'k', linestyle = ':', alpha = 0.5)
graph.plot(xp, np.full(xp.shape, 0), c = 'k', linestyle = ':', alpha = 0.5)
graph.show()


# How does it look? Were we able to beat the other network? Try out a number of different configurations to see how they perform!

# Conclusion
# -------
# 
# We've compared how different neural network architecture parameters influence accuracy performance, and we've tried to combine them in such a way that we maximise this performance.
