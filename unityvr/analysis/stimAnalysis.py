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
import ast

from matplotlib import pyplot as plt

def deriveTexVals(texDf, 
              std_filter = 3, #3*std deviation filter for removing large jumps
              diskSize = 5, #morphological disk
              round=-1, #rounding to the nearest 10th in deg/sec
              screenAboveFly = 32, #for pentagonal display with each screen dimension = 9.5*5.8 cm, in degs, +ve
              screenBelowFly = 60, #in degs, +ve,
              behindScreenAngle = 72, #angle beyond which stimulus is behind the screen
              conventionSwitch = 1 #left or right handed convention
             ):
    texDf = texDf.copy()
    texDf['stimAngle'] = (-texDf['azimuth'].values*conventionSwitch)%360-180 #convert to -180 to 180 left handed convention

    def deriveStimVel(angle, dt):
        vel = angle.diff()
        vel[np.abs(vel)>(np.nanmean(vel)+std_filter*np.nanstd(vel))] = 0
        vel = np.round(ski.morphology.closing(ski.morphology.opening(vel,np.ones(diskSize)),
                                              np.ones(diskSize))/dt,round)
        return vel

    if 'texName' in texDf.columns:
        dt = np.nanmedian(texDf.groupby('texName')['time'].diff())
        texDf['stimVel'] = texDf.groupby('texName')['stimAngle'].transform(deriveStimVel, dt)
    else:
        texDf['stimVel'] = deriveStimVel(texDf['stimAngle'])
    
    #apply morphological operation to remove unity noise and round off to the nearest 10th
    A = 1/(np.tan(screenAboveFly*np.pi/180)-np.tan(-screenBelowFly*np.pi/180))
    B = -np.tan(-screenBelowFly*np.pi/180)
    elevationToDegs = lambda e : np.arctan(e/A-B)*180/np.pi
    texDf['elevationDegs'] = elevationToDegs(texDf['elevation'].values).round(0)
    texDf['stimSpeed'] = np.abs(texDf['stimVel'])
    texDf['stimDir'] = np.sign(texDf['stimVel'])
    texDf['behindScreen'] = np.abs(texDf['stimAngle'])>=(180-behindScreenAngle/2);
    return texDf

def convertTextureVals(texDf, RF=True, divideBy = 'first'):
    if RF: 
        #elevation was mapped
        texDf['elevation'] = np.round(1-(texDf.ytex % 1),2)
    xtexpos = texDf.xtex.values.copy()
    if divideBy == 'min':
        xtexpos = (xtexpos - np.min(xtexpos))-1

    elif divideBy == 'first':
        xtexpos = xtexpos - xtexpos[0]
        
    xtexpos[xtexpos<0] = 1+xtexpos[xtexpos<0]
    
    texDf['azimuth'] = xtexpos*360
    texDf['sweepdir'] = np.sign(texDf.xtex) #right handed convention
    return texDf

def deriveVidVals(uvrDat, movieFolderPath, imageFile = 'stimGenDf.csv', sceneFile='scene1DArray.npy', shift = -90):
    movieFolder = uvrDat.vidDf['img'].str.split(r'\\').str.get(-2).unique()[-1]
    moviePath = movieFolderPath + movieFolder
    uvrDat.vidDf['filename'] = uvrDat.vidDf['img'].str.split(r'\\').str.get(-1)
    if imageFile is not None:
        stimGenDf = pd.read_csv(sep.join([moviePath,imageFile]),index_col=0)
        uvrDat.vidDf = pd.merge(uvrDat.vidDf, stimGenDf, on=['filename'])
    columnsToKeepVid = list(uvrDat.vidDf.columns)
    uvrDat.vidDf = pd.merge(uvrDat.posDf,uvrDat.vidDf,on = ['frame'],how='left').drop(columns='time_y').rename(columns={'time_x':'time'}).ffill()[columnsToKeepVid]
    if sceneFile is not None:
        uvrDat.sceneArray = np.load(sep.join([moviePath,sceneFile]))
        uvrDat.sceneArray = np.roll(uvrDat.sceneArray[:,:], shift=int(np.round(uvrDat.sceneArray.shape[-1]/360*shift)))
    return uvrDat

def mergeSplitTexDfs(df, category_col='texName', ignore_cols=['time', 'frame', 'dt'], splitStrIndex = 0):
    if splitStrIndex is not None: df[category_col] = df[category_col].str.replace('-','_').str.split('_').str[splitStrIndex]
    unique_categories = df[category_col].unique()
    dfs = []

    for category in unique_categories:
        df_category = df[df[category_col] == category].copy()

        # Drop the category column
        df_category.drop(columns=[category_col], inplace=True)

        # Rename columns except the ignored ones
        df_category.rename(columns={col: f"{col}_{category}" for col in df_category.columns 
                                    if col not in ignore_cols}, inplace=True)

        dfs.append(df_category)

    # Merge all category DataFrames on the ignored columns
    merged_df = dfs[0]
    for df_cat in dfs[1:]:
        merged_df = pd.merge(merged_df, df_cat, on=ignore_cols, how='outer')

    return merged_df

