# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 20:57:40 2018

@author: minhc
"""

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score
model = load_model('D:\Result\lungdetection-rate-001.h5')

"""
img = Image.open("F:/NAM4/HK1/IM-0005-0001.jpeg").convert('L')
img = img.resize((64,64))
imgArr = np.array(img)
# print(imgArr.shape)

imgArr = imgArr.reshape(1,imgArr.shape[0],imgArr.shape[1],1)

# print(imgArr)
# plt.imshow('imgArr')
# plt.show()

# Predicting the Test set results
y_pred = model.predict(imgArr)
print("Predict results: \n", y_pred[0])
print("Predict label: \n", np.argmax(y_pred[0]))

"""
gen = ImageDataGenerator()

test_batches = gen.flow_from_directory("../../data/chest_xray/test", model.input_shape[1:3], shuffle=False,
                                       color_mode="grayscale", batch_size=8)

p = model.predict_generator(test_batches, verbose=True)



pre = pd.DataFrame(p)
pre["filename"] = test_batches.filenames
pre["label"] = (pre["filename"].str.contains("PNEUMONIA")).apply(int)
pre['pre'] = (pre[1]>0.5).apply(int)

recall_score(pre["label"],pre["pre"])

roc_auc_score(pre["label"],pre[1])

tpr,fpr,thres = roc_curve(pre["label"],pre[1])
roc = pd.DataFrame([tpr,fpr]).T
roc.plot(x=0,y=1)