import numpy as np
import matplotlib.pyplot as plt
import os,sys
import pandas as pd
y,z=[],[]
dat=["angry","confused","cry","happy","irritated","surprised"]
for d in dat:
   dirs=os.listdir('aimage/'+d)
   i=0
   for file in dirs:
       x=plt.imread('aimage/'+d+"/"+file)
       y.append(x)
       z.append(d)
       i=i+1
   print(len(y))
   print(y[0].shape)
y=np.array(y)
z=np.array(z)
np.savez_compressed("smiled",x=y,y=z)

print(len(z))
print(len(y))
