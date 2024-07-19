import xarray as xr

from os.path import sep, exists
from os import mkdir, makedirs, getcwd

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import skimage as ski

import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, asdict

from matplotlib import pyplot as plt

def deriveTexVals(texDf, 
              std_filter = 3, #3*std deviation filter for removing large jumps
              diskSize = 5, #morphological disk
              round=-1, #rounding to the nearest 10th in deg/sec
              screenAboveFly = 32, #for pentagonal display with each screen dimension = 9.5*5.8 cm, in degs, +ve
              screenBelowFly = 60 #in degs, +ve
             ):
    texDf = texDf.copy()
    texDf['stimAngle'] = (-texDf['azimuth'].values)%360-180 #convert to -180 to 180 left handed convention
    vel = texDf['stimAngle'].diff().values #left handed convention
    vel[np.abs(vel)>(np.nanmean(vel)+std_filter*np.nanstd(vel))] = 0 #remove large jumps
    texDf['stimVel'] = np.round(ski.morphology.closing(ski.morphology.opening(vel,np.ones(diskSize)),
                                                        np.ones(diskSize))/np.nanmedian(texDf['time'].diff()),round)
    
    #apply morphological operation to remove unity noise and round off to the nearest 10th
    A = 1/(np.tan(screenAboveFly*np.pi/180)-np.tan(-screenBelowFly*np.pi/180))
    B = -np.tan(-screenBelowFly*np.pi/180)
    elevationToDegs = lambda e : np.arctan(e/A-B)*180/np.pi
    texDf['elevationDegs'] = elevationToDegs(texDf['elevation'].values).round(0)
    texDf['stimSpeed'] = np.abs(texDf['stimVel'])
    texDf['stimDir'] = np.sign(texDf['stimVel'])
    texDf['behindScreen'] = np.abs(texDf['stimAngle'])>=(180-36);
    return texDf

def convertTextureVals(texDf, RF=True):
    if RF: 
        #elevation was mapped
        texDf['elevation'] = np.round(1-(texDf.ytex % 1),1)
    texDf.xtex = texDf.xtex - texDf.xtex[0]
    xtexpos = texDf.xtex.values.copy()
    xtexpos[xtexpos<0] = 1+xtexpos[xtexpos<0]
    texDf['azimuth'] = xtexpos*360
    texDf['sweepdir'] = np.sign(texDf.xtex) #right handed convention
    return texDf