import matplotlib
import numpy as np # Dùng cho đại số tuyến tính và các phép toán đại số
import pandas as pd # tiền xử lý dữ liệu, và đọc/ghi file CSV (e.g. pd.read_csv)

import os # Dùng để thao tác với file và folder

from tensorflow.contrib.distributions.python.ops.bijectors import inline

print(os.listdir("D:/ML/sourcecode/data"))

#import thư viện
import keras # Dùng để xây dựng nên model vì nó hỗ trợ nhiều cho CNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential # Chúng ta khởi tạo model bằng Sequential sau đó dùng method add để thêm các layer.
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Softmax,Input,Flatten
from keras.optimizers import Adam,RMSprop,SGD
from keras.layers.merge import add
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import BatchNormalization
# Khai báo các layer dùng cho CNN
# Conv2D: (Convolutional Layers: chứa các layer trong mạng nơ ron tích chập) là convolution layer dùng để lấy feature từ image
# MaxPool2D: (Pooling Layers : Chứa các layer dùng trong mạng CNN.) dùng để lấy feature nổi bật(dùng max) và giúp giảm parameter khi training
# Dense: layer này sử dụng như một layer neural network bình thường.
# Dropout: layer này dùng như regularization cho các layer hạn chế overfiting https://viblo.asia/p/dropout-trong-neural-network-E375zevdlGW
# Softmax: (chọn activation function) dùng trong multi classifier
# Input:  layer này sử dụng input như 1 layer
# Flatten: dùng để lát phẳng layer để fully connection
# Activation: dùng để chọn activation.

# Dùng để chọn thuật toán training
# SGD: Stochastic gradient descent optimizer
# Adam: Adam optimizer
# RMSprop: RMSProp optimizer

# Merge Layers : chứa các layers giúp chúng ta cộng,trừ,hoặc nối các layer như các vector
# add: cộng các layers


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score
# roc_auc_score: Tính toán phần diện tích dưới Receiver Operating Characteristic Curve (ROC AUC) từ các điểm dự đoán
# roc_curve: Tính toán ROC
# recall_score: Tính toán recall
from keras.metrics import categorical_accuracy
# metrics: là thước đo để ta đánh giá accuracy của model.
# categorical_accuracy: nếu y_true==y_pre thì trả về 1 ngược lại 0,dùng cho nhiều class
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.image import ImageDataGenerator
# Image Preprocessing tiền xử lý image
# ImageDataGenerator: tạo thêm data bằng cách scale,rotation…
from tensorflow import set_random_seed
os.environ['PYTHONHASHSEED'] = "0"
np.random.seed(1)
set_random_seed(2)

# khởi tạo models
model = Sequential()
# Chúng ta khởi tạo model bằng Sequential sau đó dùng method add để thêm các layer.
# group 1
# layer 1
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same",
                 input_shape=(64,64,1)))
# Dùng convolution layer (Conv2D) dùng để lấy feature từ image, trong đó:
# filters: số filter của convolution layer
# kernel_size: size Sliding window trượt trên image
# relu: max(0,x) dùng trong các layer cnn để giảm chi phí tính toán.  
# Có tác dụng đưa các giá trị âm về thành 0. Để loại bỏ các giá trị âm 
# không cần thiết mà có thể sẽ ảnh hưởng cho việc tính toán ở các layer sau đó.
# padding="same": có sử dụng padding (="valid": không dùng)
# input_shape: chính là kích thước của dữ liệu đầu vào.
# Layer đầu tiên là layer input nên có input_shape

# layer 2
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
# layer 3
model.add(BatchNormalization())
# layer 4
model.add(MaxPooling2D(pool_size=(2,2)))
# MaxPooling2D: lấy những đặc điểm nổi bật nhất và resize lại ảnh
# pool_size: size pooling, thường có size ma trận 2x2, đối với ảnh lớn thì 4x4

# layer 5
model.add(Dropout(rate=0.25))
# Hạn chế overfiting với tỉ lệ Dropout = 0.25   

# group 2
# layer 6
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
# layer 7
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
# layer 8
model.add(BatchNormalization())
# layer 9 
model.add(MaxPooling2D(pool_size=(2,2)))
# MaxPooling2D: lấy những đặc điểm nổi bật nhất và resize lại ảnh
# layer 10
model.add(Dropout(rate=0.25))
# Hạn chế overfiting với tỉ lệ Dropout = 0.25

# layer 11
model.add(Flatten())
# Flatten: dùng để lát phẳng layer để fully connection
# chuyển thành ma trận cột

# layer 12
model.add(Dense(1024,activation="relu"))
# Dense layer này sử dụng như một layer neural network bình thường.
# 1024: chiều output (với filters ảnh đầu vào là 32x32)
# activation: dùng để chọn activation="relu"
# relu max(0,x) dùng trong các layer cnn để giảm chi phí tính toán

# layer 13
model.add(BatchNormalization())
# layer 14
model.add(Dropout(rate=0.4))
# Hạn chế overfiting với tỉ lệ Dropout = 0.4
# layer 15 output
model.add(Dense(2, activation="softmax"))
# activation: softmax dùng trong multi classifier

# learning rate = 0,0001
model.compile(Adam(lr=0.001),loss="categorical_crossentropy", metrics=["accuracy"])
###
# Sau khi build model xong thì compile nó có tác dụng biên tập lại toàn bộ model của chúng ta đã build. 
# Ở đây chúng ta có thể chọn các tham số để training model như : 
# thuật toán training thông qua tham số optimizer (ta chọn Adam để tối ưu với learning_rate = 0.0001), 
# function loss của model chúng ta có thể sử dụng mặc định hoặc tự build thông qua tham số loss, 
# (categorical_crossentropy dùng trong classifier nhiều class)
# chọn metrics hiện thị khi model được training ###

### Trong mô hình Keras với API chức năng, cần gọi fit_generator 
# để huấn luyện dữ liệu hình ảnh được tăng cường bằng cách sử dụng ImageDataGenerator.###
gen = ImageDataGenerator()
# import data và xám hóa
train_batches = gen.flow_from_directory("D:/ML/sourcecode/data/train",model.input_shape[1:3],color_mode="grayscale",shuffle=True,seed=1,
                                        batch_size=16)
valid_batches = gen.flow_from_directory("D:/ML/sourcecode/data/val", model.input_shape[1:3],color_mode="grayscale", shuffle=True,seed=1,
                                        batch_size=16)
test_batches = gen.flow_from_directory("D:/ML/sourcecode/data/test", model.input_shape[1:3], shuffle=False,
                                       color_mode="grayscale", batch_size=8)
# training dữ liệu ảnh
model.fit_generator(train_batches,validation_data=valid_batches,epochs=3)

# Save the model
model.save('D:\Result\lungdetection-rate-001.h5')

# learning rate = 0,0001 cho độ chính xác cao hơn
# model.compile(Adam(lr=0.0001),loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit_generator(train_batches,validation_data=valid_batches,epochs=3)
