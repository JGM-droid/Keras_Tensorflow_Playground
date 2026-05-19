from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

print("imports successful")

#create a sequential neural network model
model=Sequential()

print("sequential model created")

#hidden layer with 3 neurons and relu activation function
model.add(Dense(units=3, activation='relu', input_shape=(3,)))

print("hidden layer added")

# Output layer with 1 neuron
model.add(Dense(units=1))

print("output layer added")