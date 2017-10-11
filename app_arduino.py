from darkflow.net.build import TFNet
import cv2
from bboxreg import model_lev2, test_model
import tensorflow as tf
import argparse
import numpy as np
#########################################
#!/usr/bin/python
import serial
import syslog
import time

#The following line is for serial over GPIO
port = '/dev/ttyACM0'


ard = serial.Serial(port,9600)#,timeout=5)
time.sleep(3)
i = 0
setTempCar1 = 1
##########################################

options = {"model": "cfg/tiny-yolo-voc-hand.cfg", "load": -1, "threshold": 0.1, "gpu":0.99}
session = tf.InteractiveSession()

parser = argparse.ArgumentParser()
#parser.add_argument("--input_dir", help="path to folder containing images")
#parser.add_argument("--mode", default="test", choices=["train", "test", "export"])
#parser.add_argument("--output_dir", default=None, help="where to put output files")
parser.add_argument("--checkpoint", default= "check_bbox/model-2000.ckpt",help="directory with checkpoint to resume training from or use for testing")
#parser.add_argument("--textfile",default="label.txt",help="text_file containing training labels")
#parser.add_argument("--batchsize",default=8,help="Batch size for training")
a = parser.parse_args()

def bounding_box_coordinates(x, img, model):
    ckpdir=a.checkpoint
    tester=model
    saver=tf.train.Saver()
    saver.restore(session,ckpdir)
    #x_images,y_train,no_examples=load_examples_train()
    #cap.isOpened():
    #    ret,frame=cap.read()
    #    frame = cv2.cvtColor(framsaver=tf.train.Saver()
    #whilee, cv2.COLOR_BGR2GRAY)
    #cap = cv2.VideoCapture(0)
    #    frame=cv2.resize(frame,(128,128))
    #    cv2.imshow('video',frame)
    #    if cv2.waitKey(10) & 0xFF==ord('q'):
    #        break
    #    frame=2*frame/255-1
    img=np.reshape(img, (-1, 99,99,3))
    #    output=session.run(tester,feed_dict={x:frame,keep_prob:1.0})
    #    print(np.argmax(output))
    #img=np.expand_dims(np.expand_dims(img,-1),0)
    output=session.run(tester,feed_dict={x:img})
    return (output)
    
tfnet = TFNet(options)
cap = cv2.VideoCapture(0)
x = tf.placeholder(dtype = tf.float32, shape = [None, 99,99,3])
# Initializing the model
model = model_lev2(x)
bot_cont=[]
val_arr=[]
nstep=0
k0=0
nodet=0
while True:
    ret, imgcv = cap.read()
    #imgcv = cv2.imread("./sample_i
    # mg/dog.jpg")
    result = tfnet.return_predict(imgcv)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(result)==0:
        nodet+=1
        if nodet>5:
            setTempCar1 = 3
            setTemp1 = str(setTempCar1)
            ard.write(setTemp1)
    if len(result) > 0:
        nodet=0
        max_confidence = result[0]["confidence"]
        argmax = 0
        for idx,item in enumerate(result):
            if max_confidence>item["confidence"]:
                argmax = idx
                max_confidence = item["confidence"]
        
        for item in [result[argmax]]:
            cv2.rectangle(imgcv, (item["topleft"]["x"], item["topleft"]["y"]), (item["bottomright"]["x"], item["bottomright"]["y"]), (0,255,0), 2)
            cv2.putText(imgcv, str(item['confidence']),(item["topleft"]["x"], item["topleft"]["y"]) , font, 0.8, (0, 255, 0), 2)
            hand = imgcv[item["topleft"]["y"]:item["bottomright"]["y"], item["topleft"]["x"]:item["bottomright"]["x"]]
            hand = cv2.resize(hand, (99,99))
            bbox = bounding_box_coordinates(x, hand, model)
            [[x_,y_]] = np.array(bbox).astype('int')
            x_=x_*(item["bottomright"]["x"]-item["topleft"]["x"])/99
            y_=y_*(item["bottomright"]["y"]-item["topleft"]["y"])/99
            cv2.circle(imgcv, (x_ + item["topleft"]["x"], y_ + item["topleft"]["y"]), 3, (255,0,0), 3)
            fing_tip=x_ + item["topleft"]["x"]
            val_arr.append(fing_tip)
            
            if nstep>0:
                k0=k0+(val_arr[nstep]-val_arr[nstep-1])
            if (nstep)==4:
                del val_arr[:]
                if k0>0:
                    print(1)
                    setTempCar1 = 1
                    setTemp1 = str(setTempCar1)
                    ard.write(setTemp1)
                    #bot_cont.append(1)
                if k0<0:
                    setTempCar1 = 2
                    setTemp1 = str(setTempCar1)
                    ard.write(setTemp1)
                    print(2)
                    #bot_cont.append(2)
                k0=0
                nstep=0
                continue
            if(item is not None):
                nstep=nstep+1

    cv2.imshow("images", imgcv)
    cv2.waitKey(100)
'''
#!/usr/bin/python
import serial
import syslog
import time

#The following line is for serial over GPIO
port = '/dev/ttyACM0'


ard = serial.Serial(port,9600)#,timeout=5)
time.sleep(3)
i = 0
setTempCar1 = 1

while True:  
    # Serial write section
    if setTempCar1==4:
        setTempCar1=1
    setTempCar2 = 2
    setTempCar3 = 3
    
    setTemp1 = str(setTempCar1)
    setTemp2 = str(setTempCar2)
    setTemp3 = str(setTempCar3)
    print ("Python value sent: ")
    print (setTemp1)
    ard.write(setTemp1)
    setTempCar1 += 1
    #ard.flushInput()
    time.sleep(3) # with the port open, the response will be buffered 
                  # so wait a bit longer for response here
    #ard.write(setTemp2)
    #ard.flushInput()
    #time.sleep(3) # with the port open, the response will be buffered 
    #ard.write(setTemp3)
    #time.sleep(3)
    # Serial read section
    msg = ard.read(ard.inWaiting()) # read everything in the input buffer
    print ("Message from arduino: ")
    print (msg)
'''