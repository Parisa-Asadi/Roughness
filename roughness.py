# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 23:52:16 2021

@author: pza0029; Parisa Asadi
roughness. it consider the center of fracture as a base line.
 So, if your offset is other place you should modify this code. 
 The correct way of calculating raughness is having AFM image and use this 
 3D images to analyze the roughness. but this code is just for 
 one cross section of SEM image and is not accurate. Go to Text file "readMe"
 to get more inforation for roughness calculation.
"""

import cv2
import numpy as np 
import glob
import os
import pandas as pd 
import matplotlib.pyplot as plt
#########################################

os.chdir(r"C:\Users\pza0029\Box\Shared with Parisa\Olivia's paper") #main folder    

###for mass calcuation
#data_path_image = os.path.join( '*.tiff') 
#image = glob.glob(data_path_image)

###for just a limitted number of images
image = ["Mancos 1-1.tif","Marcel 1-3 copy.tiff"]

i=0 
f1=0
Results_Ra = pd.DataFrame()
Slop_final = pd.DataFrame()

for f1 in range(len(image)):
    img2= cv2.imread(image[f1],0) # 0 means grayscale, 1 BGR, -1 as its real.
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)#cv2 always read BGR so you should convert it.
    name= image[f1].split(".")[0]
    cv2.imshow('img2',img2 )
    #fracture is 1 others is zero
    segm1 = (img2 == 0)
    all_segments = np.float32(np.zeros((img2.shape[0],img2.shape[1])))
    all_segments[segm1]=1
    cv2.imshow('all_segments',all_segments)
    
    ####measure the hight of fracture
    #it is 1/N sigma(Yi) where Yi is number of 1 in each row since we want it from the center of
    #I diveded it by 2 and N is number of rows
    df = pd.DataFrame()
    df['sum'] = np.count_nonzero(all_segments == 1, axis=1)
    df['sum^2']= df['sum'].pow(2)
    Ra = 1 / (2*img2.shape[0]) * (df["sum"].sum())
    Ra1 = 1 / (2*img2.shape[0]) * (df["sum"].sum())- 1/2 * int(df.min(axis = 0)[0])
    Rq = (1 / (4*img2.shape[0]) * df['sum^2'].sum())**(1/2)
    Rq1 = ((1 / (4*img2.shape[0]) * df['sum^2'].sum()))**(1/2) - 1/2 * int(df.min(axis = 0)[0])
    Ra2 = 1 / (2*img2.shape[0]) * (df["sum"].sum())- 1/2 * int(df.mean(axis = 0)[0])
    Rq2 = ((1 / (4*img2.shape[0]) * df['sum^2'].sum()))**(1/2) - 1/2 * int(df.mean(axis = 0)[0])
    #save
    df.to_csv(f"{name}.csv",index=False)
    
    Results_Ra.loc[0,name]= Ra
    Results_Ra.loc[1,name]= Ra1
    Results_Ra.loc[2,name]= Rq
    Results_Ra.loc[3,name]= Rq1
    Results_Ra.loc[4,name]= Ra2
    Results_Ra.loc[5,name]= Rq2
    #Results_Ra.loc[1,name]= Ra1
    
    ####get the slop like peter's work
    
    all_segments1 = np.float32(np.zeros((img2.shape[0]-1,1)))
    for ii in range(0,img2.shape[0]-1):
        all_segments1[ii,0]= abs(df.iat[ii+1,0]-df.iat[ii,0])/1
        Slop = 1 / (img2.shape[0]) * np.sum(all_segments1)
    
    #####save
        Slop_final.loc[0,name]= Slop
        #Results_Ra.loc[0,name]= Ra
Results_Ra.to_csv("Results_Ra.csv",index=["Ra","Ra1","Rq","Rq1"])
Slop_final.to_csv("Slop_final.csv",index=False)   