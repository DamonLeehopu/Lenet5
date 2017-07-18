# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:06:30 2017

@author: Damon Lee
"""

from Layer import *
import numpy as np

class FCLayer(Layer):
    def __init__(self,lay_len, pre_nodesnum):
        Layer.__init__(self, [[1, lay_len]])
        Fi = pre_nodesnum + 1
        self.weight = np.random.uniform(-2.4/Fi, 2.4/Fi, [lay_len, pre_nodesnum]) # 84x120
        self.bias = np.random.uniform(-2.4/Fi, 2.4/Fi, [lay_len]) #84
        
    def fc_op(self, pre_maps, node_index) :
            pre_nodes = pre_maps.reshape([pre_maps.shape[0] * pre_maps.shape[1] * pre_maps.shape[2]]) #120
            val  = sum(self.weight[node_index] * pre_nodes) + self.bias[node_index] 
            val = np.exp((4.0/3)*val)
            self.maps[0][0][node_index] = 1.7159 *  (val -1) / (val + 1)

    def feedforward(self, pre_mapset) :
            for i in range(len(self.maps[0][0])) : 
                    self.fc_op(pre_mapset, i)

    def back_propa(self, pre_mapset, current_error, learn_rate, isweight_update) :
        self.current_error = current_error
        pcurrent_error = [((2.0/3)*(1.7159 - (1/1.7159) * self.maps[0][0][i]**2))*current_error[0][0][i]\
            for i in range(self.maps.shape[-1])]
        
        # 1x84 > 84x1 dot 1x120 > 84x120
        weight_update = np.dot(np.matrix(pcurrent_error).T, \
            np.matrix(pre_mapset.reshape([1, pre_mapset.shape[0] * pre_mapset.shape[1] * pre_mapset.shape[2]])))
        
        bias_update = np.array(pcurrent_error)

        if isweight_update :
            self.weight -= learn_rate * weight_update
            self.bias -= learn_rate * bias_update
        pre_error = np.array(np.dot(np.matrix(pcurrent_error), np.matrix(self.weight))).reshape(pre_mapset.shape)
        return pre_error                                      
                                      