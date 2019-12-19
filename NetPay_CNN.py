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

#Reshaping  & Data augmentation
TrainData = TrainData.reshape(-1, 100, 192, 1)
ValidationData = ValidationData.reshape(-1, 100, 192, 1)
TestData = TestData.reshape(-1, 100, 192, 1)

DataAugmentation = ImageDataGenerator(vertical_flip=True, horizontal_flip=True, zoom_range=0.5)
DataAugmentation.fit(TrainData, augment=True)

#CNN Architecture
Model = Sequential()
#Convolutional layers
Model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation = 'relu', padding='same', input_shape=(100,192,1)))
Model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
Model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', padding='same'))
Model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
Model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding='same'))
Model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
Model.add(Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding='same'))
Model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
Model.add(Conv2D(256, kernel_size=(3, 3), activation = 'relu', padding='same'))
Model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
Model.add(Flatten()) 
#Fully connected layers
Model.add(Dense(64, activation='relu'))    
Model.add(Dense(1, activation='linear'))

#CNN hyperarameter
adam_mod = keras.optimizers.Adam(lr=0.0001)
Model.compile(optimizer=adam_mod, loss='mean_squared_error')
Stop = EarlyStopping(monitor='val_loss', 
                    mode='min', 
                    patience=5, 
                    restore_best_weights=True)
Batch = 64
Train = Model.fit_generator(DataAugmentation.flow(TrainData, TrainLabel, batch_size=Batch), 
                            epochs=50,
                            steps_per_epoch=  TrainData.shape[0]//Batch,
                            validation_data= (ValidationData, ValidationLabel),
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
plt.title('Training and validation loss of the network')
plt.ylabel('Mean Squared Error (MSE)')
plt.xlabel('Iteration')
plt.legend()
plt.savefig('LossPlot.jpg')

#Printing elapsed time
print("Elapsed time: %.2f" % (elapsed/60),"min")