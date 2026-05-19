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

#display the model architecture summary
model.summary()

#compile the neural network model
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae'])
              
print("model compiled successfully")

# Sample housing data
# Columns: bedrooms, bathrooms, square footage
X_train = np.array([
    [0.25, 0.20, 0.30],
    [0.40, 0.35, 0.50],
    [0.60, 0.55, 0.70],
    [0.80, 0.75, 0.90],
])

# Target prices in thousands
y_train = np.array([200, 300, 400, 500])

X_test = np.array([
    [0.50, 0.50, 0.42],
    [0.30, 0.25, 0.35],
])

y_test = np.array([350, 250])

print("Training and test data created")

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=2,
    verbose=1,
    validation_data=(X_test, y_test)
)

print("Model training complete")