#!/usr/bin/env python
# coding: utf-8

# # 作業目標:
# 
#     使用CIFAR100, 數據集變大的影響
#     
#     
# # 作業重點:¶
# 
#    了解 CIFAR100 跟 CIFAR10 數據及差異
# 

# In[1]:


import numpy
from keras.datasets import cifar100
import numpy as np
np.random.seed(100)


# 
# # 資料準備
# 

# In[38]:


(x_img_train,y_label_train), (x_img_test, y_label_test)=cifar100.load_data(label_mode='fine') 


# In[39]:


print('train:',len(x_img_train))
print('test :',len(x_img_test))


# In[40]:


# 查詢檔案維度資訊
x_img_train.shape


# In[41]:


# 查詢檔案維度資訊
y_label_train.shape


# In[42]:


# 查詢檔案維度資訊
x_img_test.shape


# In[43]:


# 查詢檔案維度資訊
y_label_test.shape


# In[33]:


#針對物件圖像數據集的類別編列成字典

label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
            5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}


# In[57]:


CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


# In[62]:


#導入影像列印模組
import matplotlib.pyplot as plt

#宣告一個影像標記的函數
def plot_images_labels_prediction(images,labels,prediction,
                                  idx,num=100):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>100: num=100 
    for i in range(0, num):
        ax=plt.subplot(20,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
                
        title=str(i)+','+CIFAR100_LABELS_LIST[labels[i][0]]
        if len(prediction)>0:
            title+='=>'+CIFAR100_LABELS_LIST[prediction[i]]
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()


# In[63]:


#針對不同的影像作標記

plot_images_labels_prediction(x_img_train,y_label_train,[],0)


# In[64]:


print('x_img_test:',x_img_test.shape)
print('y_label_test :',y_label_test.shape)


# # Image normalize 

# In[65]:


x_img_train[0][0][0]


# In[66]:


x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0


# In[67]:


x_img_train_normalize[0][0][0]

轉換label 為OneHot Encoding

# In[68]:


y_label_train.shape


# In[69]:


y_label_train[:5]


# In[70]:


from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)


# In[71]:


y_label_train_OneHot.shape


# In[72]:


y_label_train_OneHot[:5]


# In[ ]:




