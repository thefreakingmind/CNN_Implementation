import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

dataframe = pd.read_csv('train.csv')
dataframe.iloc[3,1:].values.reshape(28,28).astype('uint8')

dataframe_x = dataframe.iloc[:,1:].values.reshape(len(df),28,28,1)
y = dataframe.iloc[:,0].values
dataframe_y = keras.utils.to_categorical(y, num_classes=10)

dataframe_x = np.array(dataframe_x)
dataframe_y = np.array(dataframe_y)

x_train, x_test, y_train, y_test = train_test_split(dataframe_x, dataframe_y, test_size=0.2, random_state=4)

#Neural Network
model = Sequential()
model.add(Convolution2D(32,3, data_format='channels_last', activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(10))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10,validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)
