# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:28:09 2017

@author: Damon Lee
"""

import sys
from numpy import *

class Layer(object):
    def __init__(self,lay_size=[]):
        self.lay_size=lay_size
        self.maps=[]
        for map_size in lay_size:
            self.maps.append(zeros(map_size))
        self.maps=array(self.maps)
