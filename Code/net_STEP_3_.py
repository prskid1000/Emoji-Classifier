from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils as ut
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.utils import multi_gpu_model
p=""
p=input("Enter the path where you extracted aimage folder ending with / = ")
model = Sequential()
model.add(Conv2D(32, (4, 4), input_shape=(100,100,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])
model_json=model.to_json()
with open(p+'aimage/model.json','w') as json_file:
 json_file.write(model_json)
x=np.load(p+'aimage/data.npz')
label=LabelEncoder()
labels=label.fit_transform(np.array(x['y']))
b=ut.to_categorical(labels)
model.fit(x['x'],b,epochs=5)
model.save_weights(p+"aimage/data_weights.h5")
