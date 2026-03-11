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

def findImgFrameTimes(uvrDat,imgMetadat,diffVal=3, pdAlign=False, **kwargs):
    #find the dropped frames and use those to clean up nidDf
    if pdAlign:
        uvrDat.nidDf, _ = alignWithPdSignal(uvrDat.nidDf, **kwargs)
    else:
        uvrDat.nidDf['frameToAlign'] = uvrDat.nidDf['frame'].copy()

    #find the start of each volume from the analog signal
    #now relies on upward crossing of a threshold line
    imgInd = np.where(np.diff((uvrDat.nidDf['imgfsig'].values > diffVal).astype(int)) == 1)[0]

    print('Number of imaging frames detected:', len(imgInd))
    
    imgFrame = uvrDat.nidDf.frameToAlign.values[imgInd].astype('int')

    #take only every x frame as start of volume
    volFrame = imgFrame[0::imgMetadat['fpv']]
    volFramePos = np.array([np.where(volFrame[i] == uvrDat.posDf.frame.values)[0][0] for i in range(len(volFrame)) if volFrame[i] in uvrDat.posDf.frame.values])
    #volFramePos = np.where(np.in1d(uvrDat.posDf.frame.values,volFrame, ))[0]

    return imgInd, volFramePos

def debugAlignmentPlots(uvrDat, imgMetadat, imgInd, volFramePos, lims=[0,100]):
    # figure to make some sanity check plots
    fig, axs = plt.subplots(1,3, figsize=(15,4), width_ratios=[1,1,1])

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

    # sanity check to see the difference in frame start times
    fps = imgMetadat['fpsscan'] #frame rate of scanimage
    sampling_rate = len(uvrDat.nidDf.dropna())/(uvrDat.nidDf.dropna()['time'].iloc[-1]-uvrDat.nidDf.dropna()['time'].iloc[0])
    axs[2].axvline(int(np.round(sampling_rate/fps)), color='r', linestyle='-')
    axs[2].axvline(int(np.round(sampling_rate/fps))+1, color='r', linestyle='--')
    axs[2].axvline(int(np.round(sampling_rate/fps))-1, color='r', linestyle='--')
    axs[2].hist(np.diff(imgInd))
    axs[2].axvline(np.diff(imgInd).min(), color='tab:blue', linestyle='-', alpha=0.5)
    axs[2].axvline(np.diff(imgInd).max(), color='tab:blue', linestyle='-', alpha=0.5)
    axs[2].set_title('Sanity check 3:\nCheck if all frame starts are equally spaced')
    vutils.myAxisTheme(axs[2])

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
def generateUnityExpDf(imgVolumeTimes, uvrDat, imgMetadat, suppressDepugPlot = False, dataframeAppend = 'Df', frameStr = 'frame', timeStr = 'volumes [s]', findImgFrameTimes_params={}, debugAlignmentPlots_params={}, mergeUnityDfs_params = {}):
     imgVolumeTimes = imgVolumeTimes.copy()

     unityDfs = [f for f in  uvrDat.__dataclass_fields__ if dataframeAppend in f]
     unityDfsDS = list([None]*len(unityDfs))

     #extracting volume start (unity) frames
     imgInd, volFramePos = findImgFrameTimes(uvrDat,imgMetadat,**findImgFrameTimes_params)
     volFrame = uvrDat.posDf.frame.values[volFramePos]

     #truncate volFrame assuming same start times of imaging and unity session
     lendiff = len(imgVolumeTimes) - len(uvrDat.posDf.time.values[volFramePos])
     if lendiff != 0:
          print(f'Truncated recording. Difference in length: {lendiff} imaging volumes')
          if lendiff > 0: imgVolumeTimes = imgVolumeTimes[:-lendiff]
          elif lendiff < 0: volFrame = volFrame[:lendiff]
     
     if not suppressDepugPlot: debugAlignmentPlots(uvrDat, imgMetadat, imgInd, volFramePos, **debugAlignmentPlots_params)

     #use volume start frames to downsample unityDfs
     for i,unityDfstr in enumerate(unityDfs):
            unityDf = getattr(uvrDat,unityDfstr)
            if (frameStr in unityDf):
                if len(unityDf[frameStr].unique())==len(unityDf[frameStr]):
                        volFrameId = np.array([np.where(volFrame[i] == unityDf.frame.values)[0][0] for i in range(len(volFrame)) if volFrame[i] in unityDf.frame.values])
                        # try: volFrameId = np.array([np.where(volFrame[i] == unityDf.frame.values)[0][0] for i in range(len(volFrame))])
                        # except IndexError: 
                        #     volFrameId = np.where(np.in1d(unityDf.frame.values,volFrame, ))[0]
                        #     print('errored out in :', unityDfstr) #in 1d gives true when the element of the 1st array is in the second array
                        #volFrameId = np.where(np.in1d(unityDf.frame.values,volFrame, ))[0] #in 1d gives true when the element of the 1st array is in the second array
                        framesinPos = np.where(np.in1d(uvrDat.posDf.frame.values[volFramePos], unityDf.frame.values[volFrameId]))[0] #which volume start frames of current Df are in posDf
                        unityDfsDS[i] = unityDf.iloc[volFrameId,:].copy()
                        unityDfsDS[i][timeStr] = imgVolumeTimes[framesinPos].copy() #get the volume start time for the appropriate volumes in the unity array
     
     expDf = mergeUnityDfs([x for x in unityDfsDS if x is not None],**mergeUnityDfs_params)
     return expDf

def truncateImgDataToUnityDf(imgData, expDf, timeStr = 'volumes [s]'):
    imgData = imgData[np.in1d(imgData[timeStr].values, expDf[timeStr].values,)].copy()
    return imgData


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

#take a scene and add imaging time to it

def addImagingTimeToSceneArr(sceneArr, uvrDat, imgDataTime, imgMetadat, timeStr = 'volumes [s]', sceneFrameStr = 'frames', **kwargs):
    expDf = generateUnityExpDf(imgDataTime, uvrDat, imgMetadat, timeStr=timeStr, **kwargs)
    timeSubSampled = pd.merge(sceneArr[sceneFrameStr].to_series().rename('frame').reset_index(drop=True), uvrDat.posDf[['frame', 'time']], on='frame', how = 'inner')['time']
    interpF = sp.interpolate.interp1d(expDf['time'], expDf[timeStr], fill_value='extrapolate')
    sceneArr = sceneArr.assign_coords(time = (sceneFrameStr, interpF(timeSubSampled)))
    return sceneArr

# take all the unity dataframes and add imaging time to them

def addImagingTimeToUvrDat(imgDataTime, uvrDat, imgMetadat, timeStr = 'volumes [s]', dataframeAppend = 'Df', frameStr = 'frame', generateExpDf_params = {}):
    expDf = generateUnityExpDf(imgDataTime, uvrDat, imgMetadat, timeStr=timeStr, dataframeAppend = dataframeAppend, frameStr=frameStr, **generateExpDf_params)
    interpF = sp.interpolate.interp1d(expDf['frame'], expDf[timeStr], fill_value='extrapolate')
    for f in  uvrDat.__dataclass_fields__:
        if dataframeAppend in f:
            
            unityDf = getattr(uvrDat,f)
            if frameStr in unityDf:
                unityDf[timeStr] = interpF(unityDf['frame'])
                setattr(uvrDat,f,unityDf)
    return uvrDat

def find_upticks(signal, smoothing=3):
    smoothed_signal = sp.ndimage.gaussian_filter1d(signal[~np.isnan(signal)], smoothing)
    normed_smoothed_signal = smoothed_signal-np.max(smoothed_signal)/2
    sign_changes = np.sign(normed_smoothed_signal)
    positive_zero_crossings = np.where((sign_changes[:-1] < 0) & (sign_changes[1:] > 0))[0]
    return positive_zero_crossings

# def alignWithPdSignal(nidDf, threshold=0.1, noFrameDropCorrection=True):
#     nidDf = nidDf.dropna().reset_index(drop=True).copy() #remove frames where no photodiode signal was logged
#     dips = np.where(np.diff((nidDf['pdsig'].values) > threshold) != 0)[0]
#     NcorrectionFrames = nidDf['frame'].values[dips[0]]-nidDf['frame'].values[0]+1 #the signal value for frame x will be dumped by frame x+1 (that's where +1 comes from)
#     print('Difference between first unity frame which dumps a high photodiode value and the first unity frame that starts logging photodiode values:',NcorrectionFrames)
#     nidDf['frameToAlign'] = np.clip(nidDf['frame'].copy() - NcorrectionFrames, nidDf['frame'].min(), nidDf['frame'].max())
#     validFrames = list(np.unique(nidDf['frameToAlign'].values)) if noFrameDropCorrection else list(np.unique(nidDf['frameToAlign'].values[dips]))
#     for f in np.arange(1,len(nidDf)):
#         if nidDf.loc[f,'frameToAlign'] in validFrames:
#             pass
#         else:
#             nidDf.loc[f,'frameToAlign'] = nidDf.loc[f-1,'frameToAlign']
#     return nidDf

def alignWithPdSignal(nidDf, pdThresh=0.1, pdClip = [0.04, 0.12], noFrameDropCorrection=True, supressPDAlignmentPlot = True, lims=[0,100]):
    # Drop NaNs and reset index for cleaner processing
    nidDf = nidDf.dropna(subset=['pdFilt']).reset_index(drop=True).copy()

    #clip the photodiode signal to a reasonable range
    nidDf['pdFilt'] = np.clip(nidDf['pdFilt'], pdClip[0], pdClip[1])
    
    # Find indices where pdsig crosses the threshold in either direction
    dips = np.where(np.diff(nidDf['pdFilt'] > pdThresh) != 0)[0]
    
    # Calculate frame correction based on the first crossing point
    NcorrectionFrames = nidDf['frame'].iloc[dips[0]] - nidDf['frame'].iloc[0] + 1
    print('Difference between first unity frame that starts logging photodiode values and first high photodiode frame:', NcorrectionFrames)
    
    # Align frames and apply the correction, clamping within frame range
    nidDf['frameToAlign'] = np.clip(nidDf['frame'] - NcorrectionFrames, nidDf['frame'].min(), nidDf['frame'].max())
    
    # Define valid frames depending on the frame drop correction setting
    validFrames = (np.unique(nidDf['frameToAlign']) if noFrameDropCorrection 
                   else np.unique(nidDf['frameToAlign'].iloc[dips]))
    
    # Adjust frame alignment with forward fill for invalid frames
    nidDf['frameToAlign'] = nidDf['frameToAlign'].where(nidDf['frameToAlign'].isin(validFrames)).ffill()

    if not supressPDAlignmentPlot:
        _, ax = plt.subplots(figsize=(3, 1))
        ax.plot(nidDf['pdFilt'].values, label='Photodiode Signal')
        ax.plot(dips, nidDf['pdFilt'].values[dips], 'ko')
        ax.set_xlim(lims[0], lims[1])
        vutils.myAxisTheme(ax)
    
    return nidDf, dips