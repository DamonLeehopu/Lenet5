# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:36:42 2017

@author: Damon Lee
"""

import numpy as np
from ConvNet import *
import time



def get_images(file_name):
    with open(file_name, "rb") as f:
        data=f.read()
    dt = np.dtype(np.uint32).newbyteorder('>')
    #get first 4 variables
    _, n_imgs, rows, cols = np.frombuffer(data[:16], dt, 4)
    #get all pictures' pixel
    all_pic = np.frombuffer(data[16:], np.uint8, n_imgs * rows * cols)
    #reshape to picture form
    all_pic = all_pic.reshape(n_imgs, rows, cols)    
    return all_pic    
    
def get_labels(file_name):    
    with open(file_name, "rb") as f:
        data=f.read()
    dt = np.dtype(np.uint32).newbyteorder('>')
    #get first 2 variables
    _, n_imgs = np.frombuffer(data[:8], dt, 2)
    #get all labels
    all_label = np.frombuffer(data[8:], np.uint8, n_imgs)  
    return all_label 

    
def enlarge_pic(pic):
    bigim = list(np.ones((32, 32)) * -0.1)
    for i in range(len(pic)) :
        for j in range(len(pic[0])) :
            if pic[i][j] > 0 :
                bigim[i+2][j+2] = 1.175
    im = np.array([bigim])
    return im    

    
    

    
def train_net(train_convnet,epoch, all_pic,all_label,learning_rate,pic_num=-1):
    
    #train
    if pic_num==-1:
        pic_num=len(all_pic)
    train_start_time=time.time()
    for i in range(epoch):
        print('start epoch:',i)
        for j in range(pic_num):
            im=enlarge_pic(all_pic[j])
            train_convnet.forward_p(im,all_label[j])
            train_convnet.back_p(im,all_label[j],learning_rate)
            if j%100==0:
                print('finished {}th picture'.format(j))
        print('Finished epoch:', i)    
    print ('training_time:', time.time() - train_start_time)        
            
    
def test_net(train_convnet,test_pic,test_label,test_num=-1):    
    correct_num=0
    if test_num==-1:
        test_num=len(test_label)
        
    for i in range(test_num):
        im=enlarge_pic(test_pic[i])
        train_convnet.forward_p(im,test_label[i])
        if np.argmax(train_convnet.output7.maps[0][0])==test_label[i]:
            correct_num += 1
    correct_rate=correct_num/test_num 
    print('testdata correct rate:', correct_rate)   
            





start_time=time.time()
print(start_time)
train_convnet=ConvNet()
epoch=5
learning_rate=0.0001
pic_num=5000
test_pic_num=1000

all_train_pic=get_images("train-images.idx3-ubyte")
all_train_label=get_labels("train-labels.idx1-ubyte")   
all_test_pic=get_images("t10k-images.idx3-ubyte")
all_test_label=get_labels("t10k-labels.idx1-ubyte")

print('start training')
train_net(train_convnet,epoch,all_train_pic,all_train_label,learning_rate,pic_num)
print('end training and start test')
test_net(train_convnet,all_test_pic,all_test_label,test_pic_num)
print('end test')







    