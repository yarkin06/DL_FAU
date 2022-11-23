# -*- coding: utf-8 -*-
"""
Created on Sun May 15 13:40:15 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

class Checker:
    
    def __init__(self, resolution, tile_size):
        self.tile_size = tile_size
        self.resolution = resolution
        
    def draw(self):
        amount = int(self.tile_size*2)
        a = np.zeros((amount,amount), dtype = int)
        a[self.tile_size:,:self.tile_size] = 1
        a[:self.tile_size,self.tile_size:] = 1
        num = int(self.resolution/(self.tile_size*2))
        b = np.tile(a,(num,num))
        self.output = b.copy()
        return b
           
    def show(self):
        output = self.draw()
        plt.imshow(output, cmap = "gray")
        plt.show()

class Circle:
    
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        
    def draw(self):
        res = self.resolution
        x = np.linspace(0,res-1,res).reshape(1,res)
        y = np.linspace(0,res-1,res).reshape(res,1)
        distance = np.sqrt((x - self.position[0])**2 + (y-self.position[1])**2)
        
        circ = distance <= self.radius
        self.output = circ.copy()
        return circ
    
    def show(self):
        output = self.draw()
        plt.imshow(output, cmap = "gray")
        plt.show()

class Spectrum:
    
    def __init__(self, resolution):
        self.resolution = resolution
        
    def draw(self):
        res = self.resolution
        spectrum = np.zeros([res,res, 3])
        spectrum[:,:,0] = np.linspace(0,1,res)
        spectrum[:,:,1] = np.linspace(0,1,res).reshape(res,1)     
        spectrum[:,:,2] = np.linspace(1,0,res)

        self.output = spectrum.copy()
        return spectrum
    
    def show(self):
        output = self.draw()
        plt.imshow(output)
        plt.show()

