# Functions for aligning imaging and VR data
import numpy as np
import matplotlib.pyplot as plt
from unityvr.viz import utils as vutils
import pandas as pd
from os.path import sep
import json
from unityvr.preproc import logproc
from unityvr.analysis import utils as autils
import scipy as sp

def findImgFrameTimes(uvrDat,imgMetadat,diffVal=3):

    imgInd = np.where(np.diff(uvrDat.nidDf['imgfsig'].values)>diffVal)[0]

    imgFrame = uvrDat.nidDf.frame.values[imgInd].astype('int')

    #take only every x frame as start of volume
    volFrame = imgFrame[0::imgMetadat['fpv']]
    volFramePos = np.where(np.in1d(uvrDat.posDf.frame.values,volFrame, ))[0]

    return imgInd, volFramePos
def debugAlignmentPlots(uvrDat, imgMetadat, imgInd, volFramePos, lims=[1000,1200]):
    # figure to make some sanity check plots
    fig, axs = plt.subplots(1,2, figsize=(12,4))

    # sanity check if frame starts are detected correctly from analog signal
    axs[0].plot(np.arange(0,len(uvrDat.nidDf.imgfsig.values)), uvrDat.nidDf.imgfsig, '.-')
    axs[0].plot(np.arange(0,len(uvrDat.nidDf.imgfsig.values))[imgInd],
             uvrDat.nidDf.imgfsig[imgInd], 'r.')
    axs[0].set_xlim(lims[0],lims[1])
    axs[0].set_title('Sanity check 1:\nCheck if frame starts are detected correctly')
    vutils.myAxisTheme(axs[0])

    # sanity check to see if time values align
    axs[1].plot(uvrDat.posDf.time.values[volFramePos],
                 uvrDat.nidDf.timeinterp.values[imgInd[0::imgMetadat['fpv']]].astype('int') )
    axs[1].plot(uvrDat.posDf.time.values[volFramePos],uvrDat.posDf.time.values[volFramePos],'r')
    axs[1].axis('equal')
    axs[1].set_xlim(0,round(uvrDat.posDf.time.values[volFramePos][-1])+1)
    axs[1].set_ylim(0,round(uvrDat.nidDf.timeinterp.values[imgInd[0::imgMetadat['fpv']]].astype('int')[-1])+1)
    axs[1].set_title('Sanity check 2:\nCheck that time values align well')
    vutils.myAxisTheme(axs[1])

def mergeUnityDfs(unityDfs, on = ['frame', 'time', 'volumes [s]'], interpolate=None):
    from functools import reduce
    unityDfMerged = reduce(lambda  left,right: pd.merge(left,right,on=on,
                                                how='outer'), unityDfs)
    for df in unityDfs:
        if len(df)<len(unityDfMerged):
            for c in list(df.columns):
                if c not in on:
                    if interpolate is not None:
                        interpc = sp.interpolate.interp1d(df.time.values,df[c].values,kind=interpolate,bounds_error=False,fill_value='extrapolate')
                        unityDfMerged[c] = interpc(unityDfMerged.time.values)
                        print("Interpolated ({}):".format(interpolate),c,end="; ")
    return unityDfMerged

#generate expDf in a general fashion
def generateUnityExpDf(imgVolumeTimes, uvrDat, imgMetadat, suppressDepugPlot = False, dataframeAppend = 'Df', frameStr = 'frame', findImgFrameTimes_params={}, debugAlignmentPlots_params={}, mergeUnityDfs_params = {}):
     imgVolumeTimes = imgVolumeTimes.copy()

     unityDfs = [f for f in  uvrDat.__dataclass_fields__ if dataframeAppend in f]
     unityDfsDS = list([None]*len(unityDfs))

     #extracting volume start (unity) frames
     imgInd, volFramePos = findImgFrameTimes(uvrDat,imgMetadat,**findImgFrameTimes_params)
     volFrame = uvrDat.posDf.frame.values[volFramePos]

     #truncate volFrame assuming same start times of imaging and unity session
     lendiff = len(imgVolumeTimes) - len(uvrDat.posDf.time.values[volFramePos])
     if lendiff != 0:
          print(f'Truncated recording. Difference in length: {lendiff} unity frames')
          if lendiff > 0: imgVolumeTimes = imgVolumeTimes[:-lendiff]
          elif lendiff < 0: volFrame = volFrame[:lendiff]
     
     if not suppressDepugPlot: debugAlignmentPlots(uvrDat, imgMetadat, imgInd, volFramePos, **debugAlignmentPlots_params)

     #use volume start frames to downsample unityDfs
     for i,unityDfstr in enumerate(unityDfs):
          unityDf = getattr(uvrDat,unityDfstr)
          if (frameStr in unityDf):
               if len(unityDf[frameStr].unique())==len(unityDf[frameStr]):
                    volFrameId = np.where(np.in1d(unityDf.frame.values,volFrame, ))[0] #in 1d gives true when the element of the 1st array is in the second array
                    framesinPos = np.where(np.in1d(uvrDat.posDf.frame.values[volFramePos], unityDf.frame.values[volFrameId]))[0] #which volume start frames of current Df are in posDf
                    unityDfsDS[i] = unityDf.iloc[volFrameId,:].copy()
                    unityDfsDS[i]['volumes [s]'] = imgVolumeTimes[framesinPos].copy() #get the volume start time for the appropriate volumes in the unity array
     
     expDf = mergeUnityDfs([x for x in unityDfsDS if x is not None],**mergeUnityDfs_params)
     return expDf


## combineImagingAndPosDf will be deprecated in the future
# generate combined DataFrame
def combineImagingAndPosDf(imgDat, posDf, volFramePos, timeDf=None, texDf=None, interpolateTexDf=False):
    expDf = imgDat.copy()
    lendiff = len(expDf) - len(posDf.x.values[volFramePos])
    if lendiff != 0:
        print(f'Truncated recording. Difference in length: {lendiff}')
        if lendiff > 0: expDf = expDf[:-lendiff]
        elif lendiff < 0: volFramePos = volFramePos[:lendiff]
    expDf['posTime'] = posDf.time.values[volFramePos]
    expDf['frame'] = posDf.frame.values[volFramePos]
    expDf['x'] = posDf.x.values[volFramePos]
    expDf['y'] = posDf.y.values[volFramePos]
    expDf['angle'] = posDf.angle.values[volFramePos]
    try:
        expDf['vT'] = posDf.vT.values[volFramePos]
        expDf['vR'] = posDf.vR.values[volFramePos]
        expDf['vTfilt'] = posDf.vT_filt.values[volFramePos]
        expDf['vRfilt'] = posDf.vR_filt.values[volFramePos]
    except AttributeError:
        from unityvr.analysis import posAnalysis
        posDf = posAnalysis.computeVelocities(posDf)
        expDf['vT'] = posDf.vT.values[volFramePos]
        expDf['vR'] = posDf.vR.values[volFramePos]
        expDf['vTfilt'] = posDf.vT_filt.values[volFramePos]
        expDf['vRfilt'] = posDf.vR_filt.values[volFramePos]
    if timeDf is None:
        try:
            expDf['s'] = posDf.s.values[volFramePos]
            expDf['ds'] = np.diff(expDf.s.values,prepend=0)
            expDf['dx'] = np.diff(expDf.x.values,prepend=0)
            expDf['dy'] = np.diff(expDf.y.values,prepend=0)
        except AttributeError:
            print("aligning: posDf has not been processed.")
        try:
            expDf['tortuosity'] = posDf.tortuosity.values[volFramePos]
            expDf['curvy'] = posDf.curvy.values[volFramePos]
            expDf['voltes'] = posDf.voltes.values[volFramePos]
            expDf['x_stitch'] = posDf.x_stitch.values[volFramePos]
            expDf['y_stitch'] = posDf.y_stitch.values[volFramePos]
        except AttributeError:
            print("aligning: posDf did not contain tortuosity, curvature, voltes or stitched positions")
        try:
            expDf['flight'] = posDf.flight.values[volFramePos]
        except AttributeError:
            expDf['flight'] = np.zeros(np.shape(expDf['x']))
            print("aligning: posDf did not contain flight")
        try:
            expDf['clipped'] = posDf.clipped.values[volFramePos]
        except AttributeError:
            expDf['clipped'] = np.zeros(np.shape(expDf['x']))
            print("aligning: posDf did not contain clipped")
    else:
        expDf['s'] = timeDf['s']
        expDf['ds'] = timeDf['ds']
        expDf['dx'] = timeDf['dx']
        expDf['dy'] = timeDf['dy']
        expDf['tortuosity'] = timeDf['tortuosity']
        expDf['curvy'] = timeDf['curvy']
        expDf['voltes'] = timeDf['voltes']
        expDf['x_stitch'] = timeDf['x_stitch']
        expDf['y_stitch'] = timeDf['y_stitch']
        print('aligning: derived values extracted from timeDf')
        
    if texDf is not None:
        texDfDS = alignTexAndPosDf(posDf, texDf, interpolate=interpolateTexDf).loc[volFramePos] #downsample merged texDf
        expDf = expDf.merge(texDfDS, how='outer', on=['frame'])
        print('aligning: derived values extracted from texDf')
        
    return expDf

## alignTexAndPosDf will be deprecated in the future
def alignTexAndPosDf(posDf, texDf, interpolate=None):
    refTime = posDf['time']
    unityDf = posDf.merge(texDf, on=['frame','time'], how='outer')
    columns_to_interp = list((set(texDf.columns) | set(posDf.columns)) - set(posDf.columns))
    
    if interpolate is not None:
        for c in columns_to_interp:
            interpc = sp.interpolate.interp1d(texDf['time'],texDf[c],kind=interpolate,bounds_error=False,fill_value='extrapolate')
            unityDf[c] = interpc(refTime)
    
    return unityDf[['time','frame']+columns_to_interp].copy()


def loadAndAlignPreprocessedData(root, subdir, flies, conditions, trials, panDefs, condtype, img = 'img', vr = 'uvr'):
    allExpDf = pd.DataFrame()
    for f, fly in enumerate(flies):
        print(fly)
        for c, cond in enumerate(conditions):

            for t, trial in enumerate(trials):
                preprocDir = sep.join([root,'preproc',subdir, fly, cond, trial])
                try:
                    imgDat = pd.read_csv(sep.join([preprocDir, img,'roiDFF.csv'])).drop(columns=['Unnamed: 0'])
                except FileNotFoundError:
                    print('missing file')
                    continue

                with open(sep.join([preprocDir, img,'imgMetadata.json'])) as json_file:
                    imgMetadat = json.load(json_file)

                with open(sep.join([preprocDir, vr,'metadata.json'])) as json_file:
                    uvrMetadat = json.load(json_file)

                prerotation = 0
                try: prerotation = uvrMetadat["rotated_by"]*np.pi/180
                except: pass

                uvrDat = logproc.loadUVRData(sep.join([preprocDir, vr]))
                posDf = uvrDat.posDf

                imgInd, volFramePos = findImgFrameTimes(uvrDat,imgMetadat)
                expDf = combineImagingAndPosDf(imgDat, posDf, volFramePos)

                if 'B2s' in panDefs.getPanID(cond) and condtype == '2d':
                    expDf['angleBrightAligned'] = np.mod(expDf['angle'].values-0*180/np.pi - prerotation*180/np.pi,360)
                else:
                    expDf['angleBrightAligned'] = np.mod(expDf['angle'].values-(panDefs.panOrigin[panDefs.getPanID(cond)]+prerotation)*180/np.pi,360)
                    xr, yr = autils.rotatepath(expDf.x.values,expDf.y.values, -(panDefs.panOrigin[panDefs.getPanID(cond)]+prerotation))
                    expDf.x = xr
                    expDf.y = yr
                #expDf['flightmask'] = np.logical_and(expDf.vTfilt.values < maxVt, expDf.vTfilt.values > minVt)
                expDf['fly'] = fly
                expDf['condition'] = cond
                expDf['trial'] = trial

                allExpDf = pd.concat([allExpDf,expDf])
    return allExpDf
