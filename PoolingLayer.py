# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:09:54 2017

@author: Damon Lee
"""
from Layer import *
import numpy as np


class PoolingLayer(Layer):
    
    def __init__(self, lay_size = [], pool_core_sizes = []) :
        Layer.__init__(self, lay_size)
        Fi = pool_core_sizes[0][0] * pool_core_sizes[0][1] + 1
        self.poolparas = np.random.uniform(-2.4/Fi, 2.4/Fi, [len(lay_size), 2])
        self.poolcore_sizes = np.array(pool_core_sizes)
        
    def pool_op(self, pre_map, pool_index) :
        pre_map_shape = pre_map.shape
        poolcore_size = self.poolcore_sizes[pool_index]
        for i in range(int(pre_map_shape[0] / poolcore_size[0])) :
            for j in range(int(pre_map_shape[1] / poolcore_size[1])) :
                val = self.poolparas[pool_index][0] * sum(pre_map[i*poolcore_size[0]:(i+1)*poolcore_size[0],\
                    j*poolcore_size[1]:(j+1)*poolcore_size[1]]) + self.poolparas[pool_index][1]
                val = np.exp((4.0/3)*val)
                self.maps[pool_index][i][j] = 1.7159 * (val -1) / (val + 1)

    def feedforward(self, pre_mapset) :
        for i in range(len(self.maps)) :
            self.pool_op(pre_mapset[i], i)
            
    def back_propa(self, pre_mapset, current_error, learn_rate, isweight_update) :
        self.current_error = current_error
        selfmap_line = self.maps.reshape([self.maps.shape[0] * self.maps.shape[1] * self.maps.shape[2]])
        currenterror_line = current_error.reshape([current_error.shape[0] * current_error.shape[1] * current_error.shape[2]])
        
        pcurrent_error = np.array([((2.0/3)*(1.7159 - (1/1.7159) * selfmap_line[i]**2))*currenterror_line[i]\
            for i in range(len(selfmap_line))]).reshape(self.maps.shape)
        

        weight_update = np.zeros([len(self.poolparas)])
        bias_update = np.zeros([len(self.poolparas)])
        pre_error = np.zeros(pre_mapset.shape)
        

        for i in range(self.maps.shape[0]) :
            for mi in range(self.maps.shape[1]) :
                for mj in range(self.maps.shape[2]) :
                    weight_update[i] += pcurrent_error[i][mi][mj] * sum(pre_mapset[i][mi*self.poolcore_sizes[i][0]:(mi+1)*self.poolcore_sizes[i][0], \
                        mj*self.poolcore_sizes[i][1]:(mj+1)*self.poolcore_sizes[i][1]])
                    pre_error[i][mi*self.poolcore_sizes[i][0]:(mi+1)*self.poolcore_sizes[i][0], \
                        mj*self.poolcore_sizes[i][1]:(mj+1)*self.poolcore_sizes[i][1]] =\
                        pcurrent_error[i][mi][mj] * self.poolparas[i][0]
                bias_update[i] += sum(pcurrent_error[i])
            
            if isweight_update :
                self.poolparas[:,0:1] -= learn_rate * np.matrix(weight_update).T
                self.poolparas[:,1:2] -= learn_rate * np.matrix(bias_update).T 
        return pre_error
