#hi i\'m akash kumar i\'am THE playboy

'''
tensorflow version 1.1.0+
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import collections
import numpy as np 
import glob
import argparse
import os
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", default="test", choices=["train", "test", "export"])
parser.add_argument("--output_dir", default=None, help="where to put output files")
parser.add_argument("--checkpoint", default= "check_bbox/model-2000.ckpt",help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--textfile",default="label.txt",help="text_file containing training labels")
parser.add_argument("--batchsize",default=8,help="Batch size for training")
a = parser.parse_args()

NO_OF_CLASSES=4
ITERATIONS=1200
EPOCHS=10
INP_HEIGHT=99
INP_BREADTH=99
INP_CHANNELS=3
SAVE_FREQ=200
#KEEP_PROB=0.7
def bounding_box_scaler(bbox, image):
    img = cv2.imread(a.input_dir+"/" + image)
    h, w, _ = img.shape
    bbox[0] = int(float(bbox[0])*w)
    bbox[1] = int(float(bbox[1])*h)
    bbox[2] = int(float(bbox[2])*w)
    bbox[3] = int(float(bbox[3])*h)
    return (bbox, h, w)
def load_examples_train():
    images_ = []
    labels_ = []
    i=0
    with open(a.textfile, "r+") as f:
        for line in tqdm(f.readlines()):
            words = line.strip().split("    ")
            image_name = words[0]
            bbox, h, w = bounding_box_scaler([words[1], words[2], words[3], words[4]], image_name)
            xmin = bbox[0]
            xmax = bbox[2]
            ymin = bbox[1]
            ymax = bbox[3]
            x_tip = int(float(words[5])*w)
            y_tip = int(float(words[6])*h)
            img = cv2.imread(a.input_dir+"/"+image_name)
            img = img[ymin:ymax, xmin:xmax]
            normalized_coordinates = np.array(((x_tip-xmin)/(xmax-xmin), (y_tip-ymin)/(ymax-ymin))).astype('float')
            #print(normalized_coordinates.shape)
            img = cv2.resize(img, (99,99))
            normalized_coordinates = normalized_coordinates*99
            #print(normalized_coordinates)
            images_.append(img)
            labels_.append(normalized_coordinates)
            i=i+1
        return images_, labels_,i

'''def load_examples_train():
    textf = a.textfile
    f = open(textf, 'r')
    images = []
    labels = []
    no_mat=np.zeros(10)
    for line in f:
        lines= line.split(',')
        filename, label=lines[0],lines[1]
        img=cv2.imread(os.path.join(os.getcwd(),a.input_dir,filename),0)
        img=2*img/255 -1
        img=np.expand_dims(img,-1)
        images.append(img)
        labels.append(label.rstrip())
        no_mat[int(label)]+=1
    labels=np.int32(labels)
    #print(no_mat)
    return images,labels
'''
def next_batch(num, data, labels, id_mat=None):
    if id_mat is None:
        id_mat=np.arange(len(data))
    idx = id_mat[:num]
    id_mat=collections.deque(id_mat)
    id_mat.rotate(-num)
    id_mat=np.array(id_mat)
    #print(id_mat[0])
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return id_mat,np.asarray(data_shuffle), np.asarray(labels_shuffle)


def conv(batch_input, out_channels, f_size=3,stride=1,padding="VALID"):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filters = tf.get_variable("filter", [f_size, f_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filters, [1, stride, stride, 1], padding)
        return conv

def fully_connected(batch_input,out_size=4):
    with tf.variable_scope("fc"):
        in_size=batch_input.get_shape()[1]
        weights=tf.get_variable("weights",[in_size,out_size],dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        bias=tf.get_variable("bias",[out_size],dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        wx=tf.matmul(batch_input,weights)
        fc_out=tf.nn.bias_add(wx,bias)
        return fc_out

def model_lev2(x):
    with tf.variable_scope("conv1"):
        lay1=conv(x,32,9,2,"VALID")
    
    with tf.variable_scope("conv2"):
        lay2=conv(lay1,21,4,1,"VALID")
        #lay2=tf.nn.max_pool(lay1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("conv3"):
        lay3=conv(lay2,32,3,1,"SAME")
        lay3=tf.nn.max_pool(lay3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding="VALID")
        #lay3_flat=tf.contrib.layers.flatten(lay3)
    
    with tf.variable_scope("conv4"):
        lay4=conv(lay3,64,3,1,"VALID")

    with tf.variable_scope("conv5"):
        lay5=conv(lay4,64,3,1,"VALID")
    
    with tf.variable_scope("pool6"):
        lay6=tf.nn.max_pool(lay5, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
    
    with tf.variable_scope("conv7"):
        lay7=conv(lay6,96,2,1,"VALID")

    with tf.variable_scope("conv8"):
        lay8=conv(lay7,96,2,1,"VALID")

    with tf.variable_scope("conv9"):
        lay9=conv(lay8,96,2,1,"VALID")

    with tf.variable_scope("fc1"):
        lay6_flat=tf.contrib.layers.flatten(lay6)
        lay6_fc=fully_connected(lay6_flat,160)
    
    with tf.variable_scope("fc2"):
        lay9_flat=tf.contrib.layers.flatten(lay9)
        lay9_fc=fully_connected(lay9_flat,160)

    with tf.variable_scope("final_layer"):
        lay_concat=tf.concat([lay6_fc,lay9_fc],1)
        out_lay=fully_connected(lay_concat,2)

    return out_lay

def model_train(x,y):
    out=model_lev2(x)
    return train(out,y)

def train(y_pred,y_train):
    with tf.name_scope("trainer"):
        #loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train,logits=y_pred)
        loss=tf.square(y_pred-y_train)
        loss=tf.reduce_mean(loss)
        train_step=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
        #correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.to_int64(y_train))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return loss,train_step

def test_model(x):
    a=model_lev2(x)
    return a
    
def main(epochs=EPOCHS):
    x=tf.placeholder(tf.float32,shape=[None,INP_HEIGHT,INP_BREADTH,INP_CHANNELS])
    y=tf.placeholder(tf.float32,shape=[None,2])
    #keep_prob=tf.placeholder(tf.float32)
    with tf.Session() as session:
        if (a.mode=="train"):
            x_images,y_train,no_examples=load_examples_train()
            iterate=np.int32(EPOCHS*no_examples/int(a.batchsize))
            trainer=model_train(x,y)
            saver=tf.train.Saver()
            if a.checkpoint is not None:
                ckpdir=a.checkpoint
                saver.restore(session,ckpdir)
            session.run(tf.global_variables_initializer())
            ckpdir=os.path.join(os.getcwd(),a.checkpoint)
            #saver.restore(session,ckpdir)
            save_dir=os.path.join(os.getcwd(),a.output_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir) 
            id_mat=None 
            for i in range(iterate):
                id_mat,x_batch,y_batch=next_batch(int(a.batchsize),x_images,y_train,id_mat)
                loss,_=session.run(trainer,feed_dict={x:x_batch,y:y_batch})
                print("Epoch: %d Iter: %d Loss: %g "%((i+1)//no_examples,i+1,loss))
                if (i+1)%SAVE_FREQ==0:
                    saver.save(session, os.path.join(save_dir, "model"),global_step=i+1)
                    print("Checkpoint Saved")
        elif(a.mode=="test"):
            #ckpdir=os.path.join(os.getcwd(),a.checkpoint)
            ckpdir=a.checkpoint
            tester=test_model(x)
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
            #    frame=np.expand_dims(np.expand_dims(frame,-1),0)
            #    output=session.run(tester,feed_dict={x:frame,keep_prob:1.0})
            #    print(np.argmax(output))
            #img,_,_=load_examples_train()
            #img=np.expand_dims(np.expand_dims(img,-1),0)
            output=session.run(tester,feed_dict={x:img})
            for idx,image in enumerate(img):
                cv2.circle(image, (output[idx][0], output[idx][1]), 3, (0,255,0), 3)
                cv2.imshow("Image", image)
                cv2.waitKey(1000)
                cv2.destroyAllWindows
            print(output.astype('int'))

