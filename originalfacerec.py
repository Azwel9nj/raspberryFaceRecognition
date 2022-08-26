import tkinter
from tkinter import *
from tkinter import messagebox as mess
from tkinter import ttk
import tkinter.simpledialog as tsd
import os
import cv2
import csv
import sqlite3
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import detect
#import time
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
import numpy as np
import originalfacerec


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)
    #Num_of_detections = len(get_list(detection_result))
    #print(detection_result.shape())
    #print(detection_result)
    #print(detection_result)
    #Add a check if array is empty try catch
    #foundDetection = ((detection_result.detections[0].classes[0].index))
    try:
      foundDetection = ((detection_result.detections[0].classes[0].index))
    except:
      foundDetection = 4

    print(foundDetection)  
    

    if(foundDetection == 0):      
      cap.release()
      cv2.destroyAllWindows()
      TrackImages()

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()

#tflite mask detection
def maskdetection():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='android.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))



#Functions===========================================================
conn = sqlite3.connect("facemaskattendance.db")
cursorObject = conn.cursor()
# create a table
try:
    
    cursorObject.execute("CREATE TABLE users(ID INTEGER PRIMARY KEY AUTOINCREMENT,name string,userId integer, createdOn TIMESTAMP)")
    cursorObject.execute("CREATE TABLE images(ID INTEGER PRIMARY KEY AUTOINCREMENT,imgname string, userId INTEGER, img blob, createdOn TIMESTAMP,FOREIGN KEY (userId) REFERENCES users(userId))")
    cursorObject.execute("CREATE TABLE attendance(ID INTEGER PRIMARY KEY AUTOINCREMENT, userId INTEGER, img blob, createdOn TIMESTAMP,FOREIGN KEY (userId) REFERENCES users(userId))")

    conn.commit()
except:
    pass
#AskforQUIT
def on_closing():
    if mess.askyesno("Quit", "You are exiting window.Do you want to quit?"):
        window.destroy()
#contact
def contact():
    mess._show(title="Contact Me",message="If you find anything weird or you need any help contact me on 'meetsuvariya@gmail.com'")

#about
def about():
    mess._show(title="About",message="This is a Facemask Recognition and Attendance System")

#clearbutton
def clear():
    txt.delete(0, 'end')
    txt2.delete(0, 'end')
    res = "1)Take Images  ===> 2)Save Profile"
    message1.configure(text=res)

#Check for correct Path
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

#check for haarcascade file
def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        mess._show(title='fechar file missing', message='some file is missing.Please contact me for help')
        window.destroy()

#check the password for change the password
def save_pass():
    assure_path_exists("Pass_Train/")
    exists1 = os.path.isfile("Pass_Train/pass.txt")
    if exists1:
        tf = open("Pass_Train/pass.txt", "r")
        str = tf.read()
    else:
        master.destroy()
        new_pas = tsd.askstring('Password not set', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='Null Password Entered', message='Password not set.Please try again!')
        else:
            tf = open("Pass_Train/pass.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered!', message='New password was registered successfully!')
            return
    op = (old.get())
    newp= (new.get())
    nnewp = (nnew.get())
    if (op == str):
        if(newp == nnewp):
            txf = open("Pass_Train/pass.txt", "w")
            txf.write(newp)
        else:
            mess._show(title='Error', message='Confirm new password again!!!')
            return
    else:
        mess._show(title='Wrong Password', message='Please enter correct old password.')
        return
    mess._show(title='Password Changed', message='Password changed successfully!!')
    master.destroy()

#change password
def change_pass():
    global master
    master = tkinter.Tk()
    master.geometry("400x160")
    master.resizable(False,False)
    master.title("Change Admin Password")
    master.configure(background="white")
    lbl4 = tkinter.Label(master,text='    Enter Old Password',bg='white',font=('times', 12, ' bold '))
    lbl4.place(x=10,y=10)
    global old
    old=tkinter.Entry(master,width=25 ,fg="black",relief='solid',font=('times', 12, ' bold '),show='*')
    old.place(x=180,y=10)
    lbl5 = tkinter.Label(master, text='   Enter New Password', bg='white', font=('times', 12, ' bold '))
    lbl5.place(x=10, y=45)
    global new
    new = tkinter.Entry(master, width=25, fg="black",relief='solid', font=('times', 12, ' bold '),show='*')
    new.place(x=180, y=45)
    lbl6 = tkinter.Label(master, text='Confirm New Password', bg='white', font=('times', 12, ' bold '))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tkinter.Entry(master, width=25, fg="black", relief='solid',font=('times', 12, ' bold '),show='*')
    nnew.place(x=180, y=80)
    cancel=tkinter.Button(master,text="Cancel", command=master.destroy, fg="white" , bg="#13059c", height=1,width=25 , activebackground = "white" ,font=('times', 10, ' bold '))
    cancel.place(x=200, y=120)
    save1 = tkinter.Button(master, text="Save", command=save_pass, fg="black", bg="#00aeff", height = 1,width=25, activebackground="white", font=('times', 10, ' bold '))
    save1.place(x=10, y=120)
    master.mainloop()

#ask for password
def psw():
    TrainImages()
    """
    assure_path_exists("Pass_Train/")
    exists1 = os.path.isfile("Pass_Train/pass.txt")
    if exists1:
        tf = open("Pass_Train/pass.txt", "r")
        str_pass = tf.read()
    else:
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("Pass_Train/pass.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    password = tsd.askstring('Password', 'Enter Password', show='*')
    
    if (password == str_pass):
        TrainImages()

    elif (password == None):
        pass
    else:
        mess._show(title='Wrong Password', message='You have entered wrong password')
"""
#User Registration
def insertOrUpdate(Id,Name):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
           cmd="UPDATE people SET Name=' "+str(name)+" ' WHERE ID="+str(Id)
 
    else:
         cmd="INSERT INTO people(ID,Name) Values("+str(Id)+",' "+str(name)+" ' )"
      
    conn.execute(cmd)
    conn.commit()
    conn.close()

def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM users WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile
#$$$$$$$$$$$$$
def TakeImages():
    name = (txt2.get())
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    #Make name sub directory
    parent_dir = "TrainingImage/"
    path = os.path.join(parent_dir, name)
    os.mkdir(path)
    ###Database Option Code
    """storeImage = im = open(img_name, 'rb').read()
    conn.execute("INSERT INTO images(name, img, createdOn) VALUES(?,?,?)",(img_name , sqlite3.Binary(storeImage),currentDateTime))
    print("{} written!".format(img_name))
    conn.commit()"""
    ###Database Option Code End
    serial = 0
    exists = os.path.isfile("StudentDetails/StudentDetails.csv")
    if exists:
        with open("StudentDetails/StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    else:
        with open("StudentDetails/StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()
    Id = (txt.get())
    
    currentDateTime = datetime.datetime.now()
    conn.execute("INSERT INTO users(name, userId, createdOn) VALUES(?,?,?)",(name ,Id ,currentDateTime))
    if ((name.isalpha()) or (' ' in name)):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        t_end = 1
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.05, 5)
            for (x, y, w, h) in faces:
                #currentDateTime = datetime.datetime.now()
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                
                cv2.imwrite(os.path.join(path,name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg"),
                            gray[y:y + h, x:x + w])
                cv2.imshow('Taking Images', img)
                
                #imgname = os.path.join(IMAGES_PATH,labels,labels+'.'+'{}.jpg'.format(str(uuid.uuid1())))
                img_name = os.path.join(path,name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg")
                storeImage = im = open(img_name, 'rb').read()
                
                conn.execute("INSERT INTO images(imgname, img, userId, createdOn) VALUES(?,?,?,?)",(img_name , sqlite3.Binary(storeImage),Id,currentDateTime))

                print("{} written!".format(img_name))
                conn.commit()
                # display the frame
                
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 9:
                break
            #t_end = t_end + 1
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Taken for ID : " + Id
        row = [serial, '', Id, '', name]
        with open('StudentDetails/StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=res)
    else:
        if (name.isalpha() == False):
            res = "Enter Correct name"
            message.configure(text=res)
########################################################################################
#$$$$$$$$$$$$$
def TrainImages():
    name = (txt2.get())
    parent_dir = "TrainingImage/"
    #path = os.path.join(parent_dir, name)
    check_haarcascadefile()
    assure_path_exists("Pass_Train/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels(os.path.join(parent_dir, name))
    try:
        recognizer.train(faces, np.array(ID))
    except:
        mess._show(title='No Registrations', message='Please Register someone first!!!')
        return
    recognizer.save("Pass_Train/Trainner.yml")
    res = "Profile Saved Successfully"
    message1.configure(text=res)
    message.configure(text='Total Registrations till now  : ' + str(ID[0]))

############################################################################################3
#$$$$$$$$$$$$$
def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empty face list
    faces_samples = []
    # create empty ID list
    Ids = []
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    # now looping through all the image paths and loading the Ids and the images
    for image_path in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(image_path).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        #In order to get id
        if os.path.split(image_path)[-1].split(".")[-1] !='jpg':
            continue
        image_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(imageNp)
        for (x, y, w, h) in faces:
            faces_samples.append(imageNp[y:y + h, x:x + w])
            Ids.append(image_id)
    return faces_samples, Ids
###########################################################################################
#$$$$$$$$$$$$$
def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    assure_path_exists("AttendanceImages/")
    currentDateTime = datetime.datetime.now()
    for k in tb.get_children():
        tb.delete(k)
    msg = ''
    i = 0
    j = 0
    recognizer =cv2.face.LBPHFaceRecognizer_create() 
    exists3 = os.path.isfile("Pass_Train/Trainner.yml")
    if exists3:
        recognizer.read("Pass_Train/Trainner.yml")
    else:
        mess._show(title='Data Missing', message='Please click on Save Profile to reset data!!')
        return
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    exists1 = os.path.isfile("StudentDetails/StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
    else:
        mess._show(title='Details Missing', message='Students details are missing, please check!')
        cam.release()
        cv2.destroyAllWindows()
        window.destroy()
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)
                ID = ID[1:-1]
                bb = str(aa)
                bb = bb[2:-2]
                attendance = [str(ID), '', bb, '', str(date), '', str(timeStamp)]
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                exists = os.path.isfile("Attendance/Attendance_" + date + ".csv")
                if exists:
                    with open("Attendance/Attendance_" + date + ".csv", 'a+') as csvFile1:
                        writer = csv.writer(csvFile1)
                        writer.writerow(attendance)
                    csvFile1.close()
                else:
                    with open("Attendance/Attendance_" + date + ".csv", 'a+') as csvFile1:
                        writer = csv.writer(csvFile1)
                        writer.writerow(col_names)
                        writer.writerow(attendance)
                    csvFile1.close()
                with open("Attendance/Attendance_" + date + ".csv", 'r') as csvFile1:
                    reader1 = csv.reader(csvFile1)
                    for lines in reader1:
                        i = i + 1
                        if (i > 1):
                            if (i % 2 != 0):
                                iidd = str(lines[0]) + '   '
                                tb.insert('', 0, text=iidd, values=(str(lines[2]), str(lines[4]), str(lines[6])))
                csvFile1.close()
                cam.release()
                cv2.destroyAllWindows()
                maskdetection()
                break
            else:
                Id = 'Unknown'
                bb = str(Id)
            cv2.putText(im, str(bb), (x, y + h), font, 1, (0, 251, 255), 2)
        cv2.imshow('Taking Attendance', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    exists = os.path.isfile("Attendance/Attendance_" + date + ".csv")
    if exists:
        with open("Attendance/Attendance_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(attendance)
        csvFile1.close()
    else:
        with open("Attendance/Attendance_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(col_names)
            writer.writerow(attendance)
            #img_name = "opencv_frame_{}.png".format(img_counter)
            """cv2.imwrite("AttendanceImages/ " + currentDateTime +  ".jpg",
                            gray[y:y + h, x:x + w])
            img_name = "TrainingImage/ " + currentDateTime + ".jpg"
            storeImage = im = open(img_name, 'rb').read()"""
                
            #conn.execute("INSERT INTO attendance(userId, img, createdOn) VALUES(?,?,?)",(ID, sqlite3.Binary(storeImage),Id,currentDateTime))
        csvFile1.close()
    with open("Attendance/Attendance_" + date + ".csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for lines in reader1:
            i = i + 1
            if (i > 1):
                if (i % 2 != 0):
                    iidd = str(lines[0]) + '   '
                    tb.insert('', 0, text=iidd, values=(str(lines[2]), str(lines[4]), str(lines[6])))
                    
    csvFile1.close()
    cam.release()
    cv2.destroyAllWindows()

#Front End===========================================================

window = tkinter.Tk()
window.title("Face and Facemask Recognition Based Attendance System")
window.geometry("1280x720")
window.resizable(True,True)
window.configure(background='#355454')

#Help menubar----------------------------------------------
menubar=Menu(window)
help=Menu(menubar,tearoff=0)
help.add_command(label="Change Password!",command=change_pass)
help.add_command(label="Contact Us",command=contact)
help.add_separator()
help.add_command(label="Exit",command=on_closing)
menubar.add_cascade(label="Help",menu=help)

# add ABOUT label to menubar-------------------------------
menubar.add_command(label="About",command=about)

#This line will attach our menu to window
window.config(menu=menubar)

#main window------------------------------------------------
message3 = tkinter.Label(window, text="Face and Facemask Recognition Based Attendance System" ,fg="white",bg="#355454" ,width=60 ,height=1,font=('times', 29, ' bold '))
message3.place(x=10, y=10,relwidth=1)

#frames-------------------------------------------------
frame1 = tkinter.Frame(window, bg="white")
frame1.place(relx=0.11, rely=0.15, relwidth=0.39, relheight=0.80)

frame2 = tkinter.Frame(window, bg="white")
frame2.place(relx=0.51, rely=0.15, relwidth=0.39, relheight=0.80)

#frame_headder
fr_head1 = tkinter.Label(frame1, text="Register New Student", fg="white",bg="black" ,font=('times', 17, ' bold ') )
fr_head1.place(x=0,y=0,relwidth=1)

fr_head2 = tkinter.Label(frame2, text="Student's attendance", fg="white",bg="black" ,font=('times', 17, ' bold ') )
fr_head2.place(x=0,y=0,relwidth=1)

#registretion frame
lbl = tkinter.Label(frame1, text="Enter ID",width=20  ,height=1  ,fg="black"  ,bg="white" ,font=('times', 17, ' bold ') )
lbl.place(x=0, y=55)

txt = tkinter.Entry(frame1,width=32 ,fg="black",bg="#e1f2f2",highlightcolor="#00aeff",highlightthickness=3,font=('times', 15, ' bold '))
txt.place(x=55, y=88,relwidth=0.75)

lbl2 = tkinter.Label(frame1, text="Enter Name",width=20  ,fg="black"  ,bg="white" ,font=('times', 17, ' bold '))
lbl2.place(x=0, y=140)

txt2 = tkinter.Entry(frame1,width=32 ,fg="black",bg="#e1f2f2",highlightcolor="#00aeff",highlightthickness=3,font=('times', 15, ' bold ')  )
txt2.place(x=55, y=173,relwidth=0.75)

message0=tkinter.Label(frame1,text="Follow the steps...",bg="white" ,fg="black"  ,width=39 ,height=1,font=('times', 16, ' bold '))
message0.place(x=7,y=275)

message1 = tkinter.Label(frame1, text="1)Take Images  ===> 2)Save Profile" ,bg="white" ,fg="black"  ,width=39 ,height=1, activebackground = "yellow" ,font=('times', 15, ' bold '))
message1.place(x=7, y=300)

message = tkinter.Label(frame1, text="" ,bg="white" ,fg="black"  ,width=39,height=1, activebackground = "yellow" ,font=('times', 16, ' bold '))
message.place(x=7, y=500)
#Attendance frame
lbl3 = tkinter.Label(frame2, text="Attendance Table",width=20  ,fg="black"  ,bg="white"  ,height=1 ,font=('times', 17, ' bold '))
lbl3.place(x=100, y=115)

#Display total registration----------
res=0
exists = os.path.isfile("StudentDetails/StudentDetails.csv")
if exists:
    with open("StudentDetails/StudentDetails.csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for l in reader1:
            res = res + 1
    res = (res // 2) - 1
    csvFile1.close()
else:
    res = 0
message.configure(text='Total Registrations : '+str(res))

#BUTTONS----------------------------------------------

clearButton = tkinter.Button(frame1, text="Clear", command=clear, fg="white", bg="#13059c", width=11, activebackground = "white", font=('times', 12, ' bold '))
clearButton.place(x=55, y=230,relwidth=0.29)

takeImg = tkinter.Button(frame1, text="Take Images", command=TakeImages, fg="black", bg="#00aeff", width=34, height=1, activebackground = "white", font=('times', 16, ' bold '))
takeImg.place(x=30, y=350,relwidth=0.89)

trainImg = tkinter.Button(frame1, text="Save Profile", command=psw, fg="black", bg="#00aeff", width=34, height=1, activebackground = "white", font=('times', 16, ' bold '))
trainImg.place(x=30, y=430,relwidth=0.89)

trackImg = tkinter.Button(frame2, text="Take Attendance", command=TrackImages, fg="black", bg="#00aeff", height=1, activebackground = "white" ,font=('times', 16, ' bold '))
trackImg.place(x=30,y=60,relwidth=0.89)

quitWindow = tkinter.Button(frame2, text="Quit", command=window.destroy, fg="white", bg="#13059c", width=35, height=1, activebackground = "white", font=('times', 16, ' bold '))
quitWindow.place(x=30, y=450,relwidth=0.89)



#Attandance table----------------------------
style = ttk.Style()
style.configure("mystyle.Treeview", highlightthickness=0, bd=0, font=('Calibri', 11)) # Modify the font of the body
style.configure("mystyle.Treeview.Heading",font=('times', 13,'bold')) # Modify the font of the headings
style.layout("mystyle.Treeview", [('mystyle.Treeview.treearea', {'sticky': 'nswe'})]) # Remove the borders
tb= ttk.Treeview(frame2,height =13,columns = ('name','date','time'),style="mystyle.Treeview")
tb.column('#0',width=82)
tb.column('name',width=130)
tb.column('date',width=133)
tb.column('time',width=133)
tb.grid(row=2,column=0,padx=(0,0),pady=(150,0),columnspan=4)
tb.heading('#0',text ='ID')
tb.heading('name',text ='NAME')
tb.heading('date',text ='DATE')
tb.heading('time',text ='TIME')

#SCROLLBAR--------------------------------------------------

scroll=ttk.Scrollbar(frame2,orient='vertical',command=tb.yview)
scroll.grid(row=2,column=4,padx=(0,100),pady=(150,0),sticky='ns')
tb.configure(yscrollcommand=scroll.set)

#closing lines------------------------------------------------
window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()