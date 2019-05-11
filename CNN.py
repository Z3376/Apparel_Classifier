
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense


# In[2]:


classifier=Sequential()


# In[3]:


classifier.add(Convolution2D(16,3,3,input_shape=(32,32,4),activation='relu',))
classifier.add(MaxPooling2D(pool_size=(2,2)))


# In[4]:


classifier.add(Flatten())


# In[5]:


classifier.add(Dense(output_dim=64,activation='relu'))
classifier.add(Dropout(0.1))


# In[6]:


classifier.add(Dense(output_dim=4,activation='softmax'))


# In[7]:


classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[8]:


classifier.summary()


# In[9]:


from keras.preprocessing.image import ImageDataGenerator


# In[10]:


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=1)
test_datagen=ImageDataGenerator(rescale=1./255)
train_set=train_datagen.flow_from_directory('./dataset/train',target_size=(32,32),batch_size=16,color_mode='rgba',class_mode='categorical')
test_set=test_datagen.flow_from_directory('./dataset/test',target_size=(32,32),batch_size=8,color_mode='rgba',class_mode='categorical')


# In[11]:


classifier.fit_generator(train_set,steps_per_epoch=120,epochs=5,validation_data=test_set,validation_steps=45)


# In[12]:


classifier.save('model7.h5')


# In[13]:


train_set.class_indices


# In[21]:


class_dict={0:'black_jeans',1:'blue_jeans',2:'blue_shirt',3:'red_shirt'}


# In[22]:


import numpy as np
from keras.preprocessing import image


# In[116]:


im_path='./examples/example_29.jpeg'


# In[117]:


test_img=image.load_img(im_path,target_size=(32,32),color_mode='rgba')
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)
test_img=test_img/255


# In[118]:


pred=classifier.predict(test_img)


# In[119]:


class_dict.get(np.argmax(pred))


# In[120]:


pred

