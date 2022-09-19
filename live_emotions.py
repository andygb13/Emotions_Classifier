from tensorflow import keras
from keras import models
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

#$pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

cl_path = r'C:\Users\andyg\Documents\ECE 499 - CV Project\Live Implementation\best_model40'
ob_path = r'C:\Users\andyg\Documents\ECE 499 - CV Project\Live Implementation\best.pt'

cl_model = keras.models.load_model(cl_path)

od_model = torch.hub.load('ultralytics/yolov5', 'custom', path=ob_path)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def process_img(sample):
    #input an image to do desired processing
    if len(sample.shape) > 2:
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    img = NormalizeData(sample)
    img = cv2.resize(img, (300,120))
    img = np.asarray(img)
    img = np.expand_dims(img,-1)
    img = img[None,:]
    return img

def get_eyes(img):
    results = od_model(img)
    results = results.pandas().xyxy[0]
    if len(results) == 1:
        x1 = int(results.iat[0,0])
        y1 = int(results.iat[0,1])
        x2 = int(results.iat[0,2])
        y2 = int(results.iat[0,3])
        flag = True
    else:
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        flag = False
    eyes = img[y1:y2,x1:x2]
    return x1,x2,y1,y2,eyes,flag

def find_emotion(eyes):
    eyes = process_img(eyes)
    classes = np.asarray(['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
    results = cl_model.predict(eyes)
    prediction = classes[results[0,:] == np.max(results[0,:])]
    return prediction[0]

cam = cv2.VideoCapture(0)
while True:
    if cam.isOpened():
        ret, frame = cam.read()
    else:
        print("cannot open camera")
        break

    x1,x2,y1,y2,eyes,detection = get_eyes(frame)
    if detection:
        pred = find_emotion(eyes)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,pred,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),3)
    
    cv2.imshow('Result', frame)

    #press ESC to quit
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()