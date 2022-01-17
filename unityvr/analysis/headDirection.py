import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ppatch

# Functions related to characterizing bump position .......................................
## Circular calculation utilities
def circDistAbs(angle1,angle2):
    #angles should both be in radians
    return np.pi - abs(np.pi - abs(angle1-angle2))

def circDist(angle1,angle2):
    #angles should both be in radians and equal length

    if type(angle1) != np.ndarray:
        dist = (angle1-angle2)%(np.pi*2)
        if dist>np.pi: dist = dist - 2*np.pi
    else:
        dist = (np.unwrap(angle1)-np.unwrap(angle2))%(np.pi*2)
        dist[dist>np.pi] = dist[dist>np.pi] - 2*np.pi
    return dist


def computeVectorPVA(angle, weights):
    """ Compute population vector average of angles
    """
    pva_x = np.cos(angle)*weights
    pva_y = np.sin(angle)*weights

    pva = np.vstack((sum(pva_x)/len(pva_x), sum(pva_y)/len(pva_x)))

    pvaLen = np.hypot(pva[0],pva[1])
    return pva, pvaLen


## Description of the (EB) bump related functions
def getRoiNum(df, roiname = 'slice'):
    roinames = [key for key in df.keys() if roiname in key ]
    return len(roinames)


def computePVA(locs, weights):
    """ Compute population vector average
    """
    nsteps = weights.shape[0]
    nvol = weights.shape[1]
    pva_x = np.cos(np.reshape(np.tile(locs,nvol),[nvol,nsteps])).T*weights
    pva_y = np.sin(np.reshape(np.tile(locs,nvol),[nvol,nsteps])).T*weights

    pva = np.vstack((sum(pva_x)/len(pva_x), sum(pva_y)/len(pva_x)))

    return pva


def getEBBumpPVA(df, roiname = 'slice'):
    roinames = [key for key in df.keys() if roiname in key ]
    nroi = len(roinames)

    roiArcPos = np.linspace(0, 2*np.pi, nroi+1)[:-1]
    roidat = df[roinames].values.T

    pva = computePVA(roiArcPos,roidat)

    pvaRad = np.mod(np.arctan2(pva[1,:],pva[0,:]), 2*np.pi)
    pvaLen = np.hypot(pva[0,:],pva[1,:])

    # flip to account for conversion to right-handed reference frame
    pvaRad = np.pi*2 - pvaRad

    return pvaRad, pvaLen, roiArcPos


# get max bump
def getMaxBumpPos(df, roiname='slice', order=3, window=7):
    from scipy.signal import savgol_filter
    roinames = [key for key in df.keys() if roiname in key ]

    roidat = df[roinames].values

    maxbump = savgol_filter(np.argmax(roidat,axis=1), window, order)

    # flip to account for conversion to right-handed reference frame
    return len(roinames)-maxbump


def shiftPVA(pva,offset):
    return (np.unwrap(pva) + offset)%(np.pi*2)


# Offset calculation
def findDFFPeaks(dff,radpos,dffth,minwidth=2):
    from scipy.signal import find_peaks

    # find peaks
    peaks, properties = find_peaks(dff, prominence=(None, 1),width=minwidth)

    #filter peaks to make sure they are legit
    peaks = peaks[dff[peaks]>dffth]
    peaksfilt = peaks[radpos[peaks]>-np.pi]
    peaksfilt = peaks[radpos[peaks]<=np.pi]
    peaksrad = radpos[peaksfilt]

    return peaks, peaksfilt

def getOffsetCandidates(expDf,minwidth=2, useBrightAlignedAngle=True):
    nroi = getRoiNum(expDf)
    roidat = expDf[['slice{}'.format(i+1) for i in range(nroi)]]
    tpts = len(roidat)
    dffth = roidat.to_numpy().reshape(nroi*tpts,1).mean() #+ roidat.to_numpy().reshape(nroi*tpts,1).std()/2

    rawoffset = [None] * tpts
    rawoffsetLoc = [None] * tpts
    rawoffsetDFF = [None] * tpts
    npeaks = np.zeros(tpts)
    for i in range(tpts):
        dff = roidat.loc[i,:].values
        roiArcPos = getArcRadPos(nroi)
        # pad by repeating once on each side
        dff = np.hstack([dff,dff,dff])
        radpos = np.hstack([roiArcPos-2*np.pi,roiArcPos,roiArcPos+2*np.pi])

        # flip to account for conversion to right-handed reference frame
        radpos = np.pi*2 - radpos

        # find DFF peaks
        peaks, peaksfilt = findDFFPeaks(dff,radpos,dffth,minwidth)

        #compute raw offsets and store them in a list of lists (list of frames, with list of raw offsets)
        offsets = [None] * len(peaks)
        for p in range(len(peaks)):
            if useBrightAlignedAngle:
                offsets[p] = circDist(radpos[peaks][p],expDf.angleBrightAligned.values[i]*np.pi/180)
            else:
                offsets[p] = circDist(radpos[peaks][p],expDf.angle.values[i]*np.pi/180)
        rawoffset[i] = offsets
        rawoffsetLoc[i] = radpos[peaks]
        rawoffsetDFF[i] = dff[peaks]
    return rawoffset, rawoffsetLoc, rawoffsetDFF


def getArcRadPos(nroi, min=0, max=2*np.pi):
    return np.linspace(0, 2*np.pi, nroi+1)[:-1] +(np.pi/nroi)


def getOffsetGroups(rawoffset, maxOffsetN=3, kernelfactordenom=1.5, peakwidth=1, peakheight=.05):
    from scipy import stats as sts
    from scipy.signal import find_peaks
    # Use offset candidate histogram to estimate distribution (KDE) and find peaks
    flat_list = np.asarray([item for sublist in rawoffset for item in sublist])

    #duplicate offset distribtion to avoid edge effects
    offsets4kde = np.hstack([flat_list-np.pi*2, flat_list, flat_list+np.pi*2])

    #estimate KDE on extended interval to avoid edge effects in peak detected
    samplpts = np.linspace(-2*np.pi, 2*np.pi, 2**6)
    kernel = sts.gaussian_kde(offsets4kde)
    kernel.set_bandwidth(bw_method=kernel.factor / kernelfactordenom)
    kdevals = kernel(samplpts)

    #find peaks in KDE
    kdepeaks, properties = find_peaks(kdevals,width=peakwidth,height=peakheight)

    #filter peaks to be within -pi and pi
    kdepeaks = kdepeaks[np.round(samplpts[kdepeaks],3)>-np.pi]
    kdepeaks = kdepeaks[np.round(samplpts[kdepeaks],3)<=np.pi]
    if (len(kdepeaks)>1) and (abs(circDist(np.round(samplpts[kdepeaks[0]],3),np.round(samplpts[kdepeaks[-1]],3))) < 0.1):
        kdepeaks = kdepeaks[:-1]
    kdeOffsets = np.nan*np.ones(maxOffsetN)
    kdeOffsets[:min(maxOffsetN,len(kdepeaks))] = np.round(samplpts[kdepeaks],3)

    return kdevals, samplpts, kdepeaks, kdeOffsets


def groupOffsetCandidates(rawoffset,rawoffsetLoc,rawoffsetDFF, kdeOffsets, maxOffsetN=3):
    # Classify each frame's offset computed earlier as belonging to one of the peaks in the KDE distribution
    tpts = len(rawoffset)
    # initialize raw offset array: frame x offset stats  x offset number
    # offset stats: label, value, location, peak dff,
    offsetArray = np.nan * np.ones((tpts,4, maxOffsetN))

    # convert raw offset to array, considering only unique values per frame
    for t in range(tpts):
        rawOffsetFrame = np.nan*np.ones(maxOffsetN)
        tmp = np.unique(np.round(rawoffset[t],3))
        rawOffsetFrame[:len(tmp)] = tmp[:min(maxOffsetN,len(tmp))]
        loc = rawoffsetLoc[t][np.logical_and(rawoffsetLoc[t]>=0,rawoffsetLoc[t]<2*np.pi)]
        dff = rawoffsetDFF[t][np.logical_and(rawoffsetLoc[t]>=0,rawoffsetLoc[t]<2*np.pi)]

        if len(loc)==0: continue

        # find which peak in kde the offsets correspond to
        tmp1 = np.reshape(np.tile(kdeOffsets,maxOffsetN),(maxOffsetN,maxOffsetN))
        tmp2 = np.reshape(np.tile(rawOffsetFrame,maxOffsetN),(maxOffsetN,maxOffsetN)).T
        offsetDist = abs(circDist(tmp1,tmp2))
        offsetDist[np.isnan(offsetDist)] = np.inf
        labs = offsetDist.argmin(axis=1)[:len(tmp)]

        if not len(loc) == len(labs):
            labs = labs[:len(loc)]

        for i,l in enumerate(labs):
            offsetArray[t,0,l] = l
            offsetArray[t,1,l] = rawOffsetFrame[i]
            offsetArray[t,2,l] = loc[i]
            offsetArray[t,3,l] = dff[i]
    npeaks = np.sum(np.isfinite(offsetArray[:,0,:]),axis=1)

    return offsetArray, npeaks


# Calcium traces vizualization .................................................
# Some ROI visualizations .......................................

def plotDFFheatmap(ax, df, roiname='slice', addColorbar=True,lefthanded=False):
    """
    Plot heatmap-style visualization of calcium imaging roi time series.
    We assume that calcium imaging rois are sorted in a left-handed rotational reference frame
    and flip the order to match the unity VR convention.
    """
    roinames = [key for key in df.keys() if roiname in key ]
    nroi = getRoiNum(df, roiname)

    order = np.arange(len(roinames),0,-1).astype('int')-1
    if lefthanded: order = np.arange(0,len(roinames)+1).astype('int')

    cax = ax.pcolor(df.posTime,order,df[roinames].values.T,cmap='Blues', edgecolors='face',shading='auto')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('\nROIs (n = {0})'.format(df[roinames].values.shape[1]))

    ax.set_ylim(-0.5,nroi-0.5)

    if addColorbar:
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        cbar = fig.colorbar(cax)
        cbar.set_label('$(F - F_0) / F_0$ (per ROI)')  # vertically oriented colorbar

    return ax, cax
