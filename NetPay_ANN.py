from numpy.random import seed
seed(33)

import numpy as np 
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Reshape, Dropout
from keras.models import Model, Sequential
from keras.callbacks import History
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers, optimizers
import matplotlib.pyplot as plt
import matplotlib.ticker
import time
import os
import logging
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.system('cls')

t = time.time()

#Loading the dataset
TrainData = np.load('71_73S_DataTrain.npy')
TrainLabel = np.load('71_73S_LabelTrain.npy')
print('Training data: ',TrainData.shape)

ValidationData = np.load('71_73S_DataValidation.npy')
ValidationLabel = np.load('71_73S_LabelValidation.npy')
print('Validation data: ',ValidationData.shape)

TestData = np.load('81_DataTest.npy')
TestLabel = np.load('81_LabelTest.npy')
print ('Test data: ',TestData.shape)

#Normalization of the array values
max_value1 = float(TrainData.max())
max_value2 = float(ValidationData.max())
max_value3 = float(TestData.max())
TrainData = TrainData.astype('float32') / max_value1
ValidationData = ValidationData.astype('float32') / max_value2
TestData = TestData.astype('float32') / max_value3

input_dim = TrainData.shape[1]

#ANN Architecture
Model = Sequential()
#Input layer
Model.add(Dense(256, input_shape=(input_dim,), activation='relu'))

#Hidden layer
Model.add(Dense(128, activation='relu'))

#Output layer 
Model.add(Dense(1, activation='linear'))

#ANN hyperarameter
adam_mod = keras.optimizers.Adam(lr=0.001)
Model.compile(optimizer=adam_mod, loss='mean_squared_error')
Stop = EarlyStopping(monitor='val_loss', 
                    mode='min', 
                    patience=5, 
                    restore_best_weights=True)
Batch = 256
Train = Model.fit(TrainData, TrainLabel, 
                validation_data= (ValidationData, ValidationLabel),
                batch_size=Batch, 
                epochs=50,
                shuffle = True,
                callbacks=[Stop])

Prediction = Model.predict(TestData)
np.save("Prediction.npy", Prediction)
A=np.load('Prediction.npy')
np.savetxt("Prediction.csv", A, delimiter=",")

PredictionError = Model.evaluate(TestData, TestLabel)
print('Prediction error (MSE):', PredictionError)

elapsed = time.time() - t

#Plot of loss function
loss = Train.history['loss']
val_loss = Train.history['val_loss']
plt.figure()
plt.plot(loss, 'b', label='Training loss')
plt.plot(val_loss, 'r', label='Validation loss')
locator = matplotlib.ticker.MultipleLocator(1)
plt.gca().xaxis.set_major_locator(locator)
formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
plt.gca().xaxis.set_major_formatter(formatter)
plt.ylim( (pow(10,-4),pow(10,0)) )
plt.title('Training and validation loss of the network')
plt.ylabel('Mean Squared Error (MSE)')
plt.xlabel('Iteration')
plt.legend()
plt.savefig('LossPlot.jpg')

#Printing elapsed time
print("Elapsed time: %.2f" % (elapsed/60),"min")