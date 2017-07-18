# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:08:50 2017

@author: Damon Lee
"""

import numpy as np
from OutputLayer import *
from FCLayer import *
from PoolingLayer import *
from ConvLayer import *


class ConvNet(object):
    def __init__(self,input_size=[]):
        
        cov3_core_sizes = [[3, 5, 5]] * 6
        cov3_core_sizes.extend([[4, 5, 5]] * 9)
        cov3_core_sizes.extend([[6, 5, 5]])
        
        cov3_mapcombindex = [[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,0],[5,0,1],\
                [0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,0],[4,5,0,1],[5,0,1,2],[0,1,3,4],[1,2,4,5],[0,2,3,5],[0,1,2,3,4,5]]
        
        
        self.convlay1=ConvLayer([[28,28]]*6,[[1,5,5]]*6)
        self.poollay2=PoolingLayer([[14,14]]*6,[[2,2]]*6)
        self.convlay3=ConvLayer([[10,10]]*16,cov3_core_sizes,cov3_mapcombindex)
        self.poollay4=PoolingLayer([[5,5]]*16,[[2,2]]*16)
        self.convlay5=ConvLayer([[1,1]]*120,[[16,5,5]]*120)
        self.fclayer6=FCLayer(84,120)
        self.output7=OutputLayer(10,84)
        
    def forward_p(self,pic,label):
        
        self.convlay1.feedforward(pic)
        self.poollay2.feedforward(self.convlay1.maps)
        self.convlay3.feedforward(self.poollay2.maps,True)
        self.poollay4.feedforward(self.convlay3.maps)
        self.convlay5.feedforward(self.poollay4.maps)
        self.fclayer6.feedforward(self.convlay5.maps)
        self.output7.softmax(self.fclayer6.maps)

    def back_p(self,pic,label,learn_rate):

        output_error = np.zeros([1, 1, 10])
        output_error[0][0][label] = 1
        
        #fclayer_error = self.outputlay7.back_propa(self.fclay6.maps, output_error, learn_rate, True)
        fclayer_error = self.output7.back_propa_softmax(self.fclayer6.maps, output_error, learn_rate, True)
        conv5_error = self.fclayer6.back_propa(self.convlay5.maps, fclayer_error, learn_rate, True)
        pool4_error = self.convlay5.back_propa(self.poollay4.maps, conv5_error, learn_rate, True)
        conv3_error = self.poollay4.back_propa(self.convlay3.maps, pool4_error, learn_rate, True)
        pool2_error = self.convlay3.back_propa(self.poollay2.maps, conv3_error, learn_rate, True)
        conv1_error = self.poollay2.back_propa(self.convlay1.maps, pool2_error, learn_rate, True)
        ilayer_error = self.convlay1.back_propa(pic, conv1_error, learn_rate, True)




    def print_weights(self):  
        pass
        