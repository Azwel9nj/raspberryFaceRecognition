#!/usr/bin/env python
# coding: utf-8

# In[1]:


WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = 'annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = 'Model/pipeline.config'
CHECKPOINT_PATH = 'Model/'


# In[2]:


import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


# In[3]:


import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


# In[4]:


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-101')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# In[5]:


import cv2 
import numpy as np


# In[6]:


category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')


# In[7]:


# Setup capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# In[8]:


# Import opencv
import cv2 

# Import uuid
import uuid

# Import Operating System
import os

# Import time
import time


# In[9]:


IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')


# In[10]:


labels = ['allCapturedImages']


# In[11]:


#if not os.path.exists(IMAGES_PATH):
#   if os.name == 'posix':
#        get_ipython().system('mkdir -p {IMAGES_PATH}')
#    if os.name == 'nt':
#         get_ipython().system('mkdir {IMAGES_PATH}')
#for label in labels:
#    path = os.path.join(IMAGES_PATH, label)
#    if not os.path.exists(path):
#        get_ipython().system('mkdir {path}')


# In[ ]:


while True: 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
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

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    number_imgs = 6
    #for imgnum in range(number_imgs):
        #imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        #cv2.imwrite(imgname, frame)
        #time.sleep(5)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break


# In[ ]:




