#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import tensorflow as tf
import numpy as np
import os
import cv2

from keras_preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory


# # Loading the pre-trained cascades

# In[2]:


print (os.getcwd())
os.chdir('C:\\Users\\Ibrahim Hameem\\Desktop\\Machine Learning\\7. Neural Nets\\Convolutional Neural Network\\Computer_Vision_A_Z_Template_Folder\\Module_1_Face_Recognition')
print ('')
print (os.getcwd())


# In[3]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# # Loading a custome pre-trained version of MobileNet V2 for image classification

# In[4]:


#Change the working directory to the location within your computer, where the pre-trained MobileNet V2 is saved 
os.chdir('C:\\Users\\Ibrahim Hameem\\Desktop\\Machine Learning\\7. Neural Nets\\Convolutional Neural Network\\Project Face Mask')
print(os.getcwd())


# In[5]:


#Load the pre-trained model
base_model = tf.keras.models.load_model('mask_model_pre-trained_1.h5')


# In[6]:


#We lock the models, such that the imported model is not trainable
base_model.trainable = False


# # Core algorithm

# ## Definining a detection function

# In[7]:


Mask_dict = {'No Mask or Incorrectly masked':1, 'Mask':0}
Color_dict = {1:(0,0,255), 0:(0,255,0)}
prediction_threshold = 0.3


# In[8]:


def maskdetect (gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 7)
    for (x,y,w,h) in faces:
        roi_color = frame[y-60:y-60+h+120,x-15:x-15+w+30]
        
        resized = cv2.resize(roi_color,(224,224))
        test_image = image.img_to_array(resized)
        test_image = np.expand_dims(test_image, axis = 0)
        result = base_model.predict(test_image)
    
        if result[0][0] >= prediction_threshold:
            prediction = 'No Mask or Incorrectly masked'
        else:
            prediction = 'Mask'
        
        frame = cv2.rectangle(frame, (x-20,y-70), (x-20+w+40, y-70+h+110),Color_dict[Mask_dict[prediction]] ,3)
        frame = cv2.rectangle(frame,(x-110,y-90), (x-110+w+220, y-130),Color_dict[Mask_dict[prediction]],-1)
        frame = cv2.putText(frame,prediction, (x-100, y-100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        
        if prediction == 'No Mask or Incorrectly masked':
            frame = cv2.putText(frame,str(np.round(result[0][0]*100,2)) + '%', (x+200, y-100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        else:
            frame =cv2.putText(frame,str(np.round((1-result[0][0])*100,2)) + '%', (x+200, y-100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
         
    return frame     


# In[9]:


video_capture = cv2.VideoCapture(0)

while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = maskdetect(gray, frame)
    cv2.imshow('Video', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break
        
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




