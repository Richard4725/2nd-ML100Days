#!/usr/bin/env python
# coding: utf-8

# # 課程目標
# 
# Keras:
# 
# 1. Keras 架構
# 2. 如何安裝
# 3. 後臺設定
# 

# In[9]:


import keras
from keras import backend as K
from keras.layers import Layer


# # 範例重點
# 
# 1. 如何使用Keras 設定後台
# 
# 2. 如何配置 CPU/GPU
# 
# 3. 常見錯誤處理
# 
# 

# In[10]:


print(keras.__version__)


# In[11]:


#  GPU加速测试, True - Windows用戶得到True也沒有關係，因為Anaconda中已經內置了MKL加速庫
import numpy 
id(numpy.dot) == id(numpy.core.multiarray.dot) 


# In[12]:


#檢查Keras float 
K.floatx()


# In[13]:


#設定浮點運算值
K.set_floatx('float16')
K.floatx()


# # **常見錯誤處理
# 
# **常見錯誤：**FutureWarning: Conversion of the second argument of issubdtype from floatto np.floatingis deprecated. In future, it will be treated as np.float64 == np.dtype(float).type.
# from ._conv import register_converters as _register_converters
# **解決方案：**pip install h5py==2.8 .0rc1，安裝h5py，用於模型的保存和載入
# 
# **切換後端**Using TensorFlow backend.
# 但是keras的backend同時支持tensorflow和theano.
# 並且默認是tensorflow,
# 
# **常見錯誤：**TypeError: softmax() got an unexpected keyword argument 'axis'
# **解決方案：**pKeras與tensorflow版本不相符，盡量更新最新版本：pip install keras==2.2
# 
# 

# In[14]:


from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Input, Dense


a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)

config = model.get_config()
print(config)


# In[ ]:




