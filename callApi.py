from multiprocessing.connection import wait
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import threading
import tensorflow as tf
from object_detection.utils import config_util
import cv2 
import numpy as np
import uuid
import sqlite3
import time
import pandas as pd
import datetime
import threading
from flask import Flask, request, Response, jsonify, send_from_directory, abort
# Setup capture

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = 'annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = 'Model/pipeline.config'
CHECKPOINT_PATH = 'Model/'
IMAGES_PATH = os.path.join('collectedimages')
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
label = ['allCapturedImages']

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
labels = ['allCapturedImages']

import base64
import json                    

import requests

api = 'http://192.168.0.129:5000/imagedetection'
#image_file = 'sample_image.png'

with open(image_file, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8")

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
  
payload = json.dumps({"image": im_b64, "other_key": "value"})
response = requests.post(api, data=payload, headers=headers)
try:
    data = response.json()     
    print(data)                
except requests.exceptions.RequestException:
    print(response.text)

img_counter = 0
while True: 
    ret, frame = cap.read()
    image_np = np.array(frame)    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections    
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64) 

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

    if detections['detection_classes'][0] == 2:        
        currentDateTime = datetime.datetime.now()
        # SPACE pressed
        #imgname = os.path.join(IMAGES_PATH,labels,labels+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        storeImage = im = open(img_name, 'rb').read()
        conn.execute("INSERT INTO images(name, img, createdOn) VALUES(?,?,?)",(img_name , sqlite3.Binary(storeImage),currentDateTime))
        print("{} written!".format(img_name))
        conn.commit()
        img_counter += 1
        try: 
            os.remove(img_name)
        except: pass
        time.sleep(7)
               
    cv2.imshow('facemask detection',  cv2.resize(image_np_with_detections, (800, 600)))       
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
    if cv2.getWindowProperty('facemask detection', cv2.WND_PROP_VISIBLE) <1:
        break