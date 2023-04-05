#!/usr/bin/env python
# coding: utf-8

# In[3]:


import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

image_folder = "C:/Users/Dell/Documents"
emotions = ["Smiling", "Sleeping", "Neutral", "Others", "Bored"]
num_images_per_emotion = 5
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    landmarks = ["Emotion"]
    for emotion in emotions:
        for i in range(1, num_images_per_emotion + 1):
            image_path = os.path.join(image_folder, f"{emotion}{i}.jpg")
            image = cv2.imread(image_path)
                
            # Resize image
            image = cv2.resize(image, (512, 512))
            
            # Recolor Feed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                     mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

            # Add emotion label to landmarks array
            coordinate = len(results.face_landmarks.landmark)
            row = [emotion]
            for val, landmark in enumerate(results.face_landmarks.landmark):
                row += [landmark.x, landmark.y, landmark.z]
                landmarks += ['x{}_{}'.format(val, emotion), 'y{}_{}'.format(val, emotion), 'z{}_{}'.format(val, emotion), 'v{}_{}'.format(val, emotion)]

            # Write landmarks data to CSV file
            with open('Data.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            cv2.imshow('Image', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            else:
                cv2.waitKey(100000)
cv2.destroyAllWindows()


# In[4]:


import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[5]:


df = pd.read_csv('Data.csv')


# In[6]:


x = df.drop('Smiling',axis = 1)
y = df['Smiling']


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)


# In[8]:


y_test


# In[9]:


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score


# In[10]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# In[11]:


model = Sequential()
model.add(Dense(units = 128, activation='relu', input_dim = len(x_train.columns)))
model.add(Dense(units = 64, activation='relu'))
model.add(Dense(units = 5, activation='softmax'))


# In[12]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')


# In[13]:


model.fit(x_train, y_train, epochs = 100, validation_data = (x_test, y_test))


# In[14]:


model.predict(x_test.iloc[[0]])


# In[15]:


model.evaluate(x_test, y_test)


# In[16]:


model.save('Detection')


# In[17]:


loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)


# In[ ]:




