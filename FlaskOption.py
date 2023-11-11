#!/usr/bin/env python
# coding: utf-8

# In[1]:

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

# Initialize Flask application
app = Flask(__name__)

# connect to database
conn = sqlite3.connect("facemaskdb.db")
cursorObject = conn.cursor()
# create a table
try:
    cursorObject.execute("CREATE TABLE images(ID INTEGER PRIMARY KEY AUTOINCREMENT,name string, img blob, createdOn TIMESTAMP)")
    conn.commit()
except:
    pass
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-701')).expect_partial()

# API that returns image with detections on it
@app.route('/imagedetection', methods= ['POST'])
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections




# In[ ]:
if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)



