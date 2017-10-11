from darkflow.net.build import TFNet
from optical_flow import optical_smooth as osmooth
import cv2
from bboxreg import model_lev2, test_model
import tensorflow as tf
import argparse
import numpy as np

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
f_count=0
det_cnt=0
while True:
    ret, imgcv = cap.read()
    if f_count==0:
        prev=imgcv
    #imgcv = cv2.imread("./sample_img/dog.jpg")
    result = tfnet.return_predict(imgcv)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(result) > 0:
        max_confidence = result[0]["confidence"]
        argmax = 0
        for idx,item in enumerate(result):
            if max_confidence>item["confidence"]:
                argmax = idx
                max_confidence = item["confidence"]
        imgcv2=imgcv
        
        for item in [result[argmax]]:
            det_cnt=det_cnt+1
            cv2.rectangle(imgcv, (item["topleft"]["x"], item["topleft"]["y"]), (item["bottomright"]["x"], item["bottomright"]["y"]), (0,255,0), 2)
            
            cv2.putText(imgcv, str(item['confidence']),(item["topleft"]["x"], item["topleft"]["y"]) , font, 0.8, (0, 255, 0), 2)
            hand = imgcv[item["topleft"]["y"]:item["bottomright"]["y"], item["topleft"]["x"]:item["bottomright"]["x"]]
            hand = cv2.resize(hand, (99,99))
            bbox = bounding_box_coordinates(x, hand, model)
            if f_count>0 and det_cnt>1:
                n_bbox=osmooth(prev,imgcv2,x_l,y_l,x_r,y_r,dt=1/7.0)
                cv2.rectangle(imgcv,(n_bbox[0],n_bbox[1]),(n_bbox[2],n_bbox[3]),(255,0,0),2)            
            x_l=item["topleft"]["x"]
            x_r=item["bottomright"]["x"]
            y_l=item["topleft"]["y"]
            y_r=item["bottomright"]["y"]
            [[x_,y_]] = np.array(bbox).astype('int')
            x_=x_*(item["bottomright"]["x"]-item["topleft"]["x"])/99
            y_=y_*(item["bottomright"]["y"]-item["topleft"]["y"])/99
            cv2.circle(imgcv, (x_ + item["topleft"]["x"], y_ + item["topleft"]["y"]), 3, (255,0,0), 3)
            prev=imgcv2
    cv2.imshow("images", imgcv)
    cv2.waitKey(100)
    f_count=f_count+1
    #print(result)
'''

import serial
import time

print("Start")
port = 7 #This will be different for various devices and on windows it will probably be a COM port.
bluetooth=serial.Serial(port, 9600)#Start communications with the bluetooth unit
print("Connected")
bluetooth.flushInput() #This gives the bluetooth a little kick
for i in range(5): #send 5 groups of data to the bluetooth
	print("Ping")
	bluetooth.write(b"BOOP "+str.encode(str(i)))#These need to be bytes not unicode, plus a number
	input_data=bluetooth.readline()#This reads the incoming data. In this particular example it will be the "Hello from Blue" line
	print(input_data.decode())#These are bytes coming in so a decode is needed
	time.sleep(0.1) #A pause between bursts
bluetooth.close() #Otherwise the connection will remain open until a timeout which ties up the /dev/thingamabob
print("Done")

'''