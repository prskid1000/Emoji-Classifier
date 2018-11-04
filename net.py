from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.utils import np_utils as ut
import numpy as np
import matplotlib.pyplot as plt
import os
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

x=np.load('smiled.npz')
label=LabelEncoder()
labels=label.fit_transform(np.array(x['y']))
b=ut.to_categorical(labels)
model.load_weights("smile_weights.h5")
model.fit(x['x'],b,epochs=5)
model.save_weights("smile_weights.h5")
p=model.predict(x['x'])
dat=["angry","confused","cry","happy","irritated","surprised"]
i=0
r=0
for d in dat:
   dirs=os.listdir('aimage/'+d)
   for file in dirs:
       n=plt.imread('aimage/'+d+"/"+file)
       n = n.reshape((1,) + n.shape)
       i+=1
       if d==dat[np.argmax(model.predict(n))]:
           r+=1
print(r/i)
