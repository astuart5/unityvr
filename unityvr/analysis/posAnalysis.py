import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
import json

from unityvr.viz import viz
from unityvr.analysis.utils import carryAttrs, getTrajFigName
from unityvr.analysis import align2img

from os.path import sep, exists, join

##functions to process posDf dataframe

#obtain the position dataframe with derived quantities
def position(uvrDat, derive = True, rotate_by = None, plot = False, plotsave=False, saveDir=None, computeVel=False, **computeVelocitiesKwargs):
    ## input arguments
    # set derive = True if you want to compute derived quantities (ds, s, dTh (change in angle), radangle (angle in radians(-pi,pi)))
    # rotate_by: angle (degrees) by which to rotate the trajectory to ensure the bright part of the panorama is at 180 degree heading.
    
    posDf = uvrDat.posDf.copy() #NOT INPLACE

    #rotate
    if rotate_by is not None:
        
        #rotate the trajectory
        posDf['x'], posDf['y'] = rotation_deg(posDf['x'],posDf['y'],rotate_by)

        posDf['angle'] = (posDf['angle']+rotate_by)%360
        uvrDat.metadata['rotated_by'] = (uvrDat.metadata['rotated_by']+rotate_by)%360 if ('rotated_by' in uvrDat.metadata) else (rotate_by%360)

    #add dc2cm conversion factor
    posDf.dc2cm = 10

    if derive:
        posDf = posDerive(posDf)
        if computeVel:
            posDf = computeVelocities(posDf,**computeVelocitiesKwargs)
        
        #get flight and clipped from flightDf dataframe
        #why? 
        if hasattr(uvrDat,'flightDf'):
            if ('flight' in uvrDat.flightDf):
                posDf['flight'] = uvrDat.flightDf['flight'].copy()
            if ('clipped' in uvrDat.flightDf):
                posDf['clipped'] = uvrDat.flightDf['clipped'].copy()

    if plot:
        fig, ax = viz.plotTrajwithParameterandCondition(posDf, figsize=(10,5), parameter='angle')
        if plotsave:
            fig.savefig(getTrajFigName("walking_trajectory",saveDir,uvrDat.metadata))

    return posDf

def posDerive(inDf):
    posDf = inDf.copy() #NOT INPLACE
    posDf['dx'] = np.diff(posDf['x'], prepend=0) #allocentric translation vector x component
    posDf['dy'] = np.diff(posDf['y'], prepend=0) #allocentric translation vector y component
    posDf['ds'] = np.sqrt(posDf['dx']**2+posDf['dy']**2) #magnitude of the translation vector
    posDf['s'] = np.cumsum(posDf['ds']) #integrated pathlength from start
    posDf['dTh'] = (np.diff(posDf['angle'],prepend=posDf['angle'].iloc[0]) + 180)%360 - 180
    posDf['radangle'] = ((posDf['angle']+180)%360-180)*np.pi/180

    #derive forward and side velocities: does not depend on rotation of trajectory and angle
    if 'dx_ft' not in posDf:
            xy = np.diff(posDf[['x', 'y']], axis=0)
            rotation_mats = np.array([np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]]).T for theta in np.deg2rad(posDf['angle'])[:-1]])
            posDf['dx_ft'], posDf['dy_ft'] = np.vstack([[0,0],np.einsum('ijk,ik->ij', rotation_mats, xy)]).T #in egocentric frame, units: decimeters, #dx_ft is forward motion (forward is positive), dy_ft is side motion (rightward is positive)
    return posDf

#segment flight bouts
def flightSeg(posDf, thresh, freq=120, plot = False, freq_content = 0.5, plotsave=False, saveDir=None, uvrDat=None):

    df = posDf.copy()

    #get spectrogram
    f, t, F = sp.signal.spectrogram(df['ds'], freq)

    spec_row = round(freq_content*len(f)*2/freq) #freq_content is in Hertz

    # 2nd row of the spectrogram seems to contain sufficient information to segment flight bouts
    flight = sp.interpolate.interp1d(t,F[spec_row,:]>thresh, kind='nearest', bounds_error=False, fill_value=0)
    
    df['flight'] = flight(df['time'])

    #carry attributes
    df = carryAttrs(df,posDf)

    if plot:
        fig0, ax0 = plt.subplots()
        ax0.plot(t,F[spec_row,:],'k');
        ax0.plot(df['time'],df['flight']*F[spec_row,:].max(),'r',alpha=0.2);
        ax0.set_xlabel("time"); plt.legend(["power in HF band","thresholded"])

        fig1, ax1 = viz.plotTrajwithParameterandCondition(df, figsize=(10,5), parameter='angle',
                                                        condition = (df['flight']==0))
        if plotsave:
            fig0.savefig(getTrajFigName("FFT_flight_segmentation",saveDir,uvrDat.metadata))
            fig1.savefig(getTrajFigName("walking_trajectory_segmented",saveDir,uvrDat.metadata))

    return df

#clip the dataframe
def flightClip(posDf, minT = 0, maxT = 485, plot = False, plotsave=False, saveDir=None, uvrDat=None):

    df = posDf.copy()

    #clip the position values according to the minT and maxT
    df['clipped'] = ((posDf['time']<=minT) | (posDf['time']>=maxT)).astype('float')

    #carry attributes
    df = carryAttrs(df,posDf)

    if plot:
        fig, ax = viz.plotTrajwithParameterandCondition(df, figsize=(10,5), parameter='angle',
                                                        condition = (df['clipped']==0))
        if plotsave:
            fig.savefig(getTrajFigName("walking_trajectory_clipped",saveDir,uvrDat.metadata))

    return df

#rotation matrix for an Nx2 vector with [x,y]
def rotation_mat_rad(theta):
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return R

#rotate the trajectory by angle theta
def rotation_deg(x,y,theta):
    theta = np.pi/180*theta
    r = np.matmul(np.array([x,y]).T,rotation_mat_rad(theta))
    return r[:,0], r[:,1]

"""
ComputeVelocities
@author: hulseb, modifed by chitniss
"""

# Add derrived quantities: Velocities
def computeVelocities(inDf, ToFilter = True,  BoxCarWidth = 9):
    posDf = inDf.copy()

    ##########################################################################################################
    # Step 1: Compute translational velocities and travel direction in allocentric coordinates.
    ##########################################################################################################

    posDf['dx_world'] = posDf.dx #one sided difference computed in position
    posDf['dy_world'] = posDf.dy #one sided difference
    #window and order for filter

    # Compute the worlds's translational speed 
    posDf['vT_world']  =  np.hypot( posDf['dx_world'], posDf['dy_world'])*(1/posDf.dt) # in distance units/s (if enforce_cm in logproc then in cm/s)

    #here we use one sided difference for dx and dy. Note that fictrac provides changes in fly's rotational velocity as one sided differences. 
    #if we want the translational speed to be the same in egocentric and allocentric coordinates, we should be using one sided differences for allocentric changes


    # Optionally filter the traces above
    if ToFilter:
        # BoxCarWidth = 9, so center sample and 3 on either side. This means they'll be
        # no overlap between samples when we add the velocity information to downsampled
        # dataframes like expDf, which are at ~10 Hz instead of ~120 Hz.  
        posDf['dx_world']      =  posDf['dx_world'].rolling(window = BoxCarWidth, center = True, min_periods=1).mean()
        posDf['dy_world']      =  posDf['dy_world'].rolling(window = BoxCarWidth, center = True, min_periods=1).mean()
        posDf['vT_world']      =  posDf['vT_world'].rolling(window = BoxCarWidth, center = True, min_periods=1).mean()
        
    # Compute the translational velocity angle in world coordinates (0 degrees is y axis. 90 degrees is positive x axis)
    # Note: we compute vT_world_angle after the filtering step above because it's a circular variable so annoying to filter after computing it.
    posDf['vT_world_angle']  =  np.arctan2( posDf['dy_world'], posDf['dx_world'] )/2/np.pi*360  # 0 degrees is y axis since arctan2(1,0) is 90 degrees. 
    #TODO: verify that this convention works with the zero defined by the brightest side. 

    # Compute the world's rotational velocity
    posDf['dangle_world']  =  np.rad2deg(np.diff( np.unwrap(np.deg2rad(posDf.angle.values)), prepend=0)) # in degrees
    posDf['vR_world']      =  posDf['dangle_world'] * (1/posDf.dt) # in degrees/second
    
    # Optionally filter the traces above
    if ToFilter:
        posDf['dangle_world']  =  posDf['dangle_world'].rolling(window = BoxCarWidth, center = True, min_periods=1).mean()
        posDf['vR_world']      =  posDf['vR_world'].rolling(window = BoxCarWidth, center = True, min_periods=1).mean()

    ##########################################################################################################
    # Step 2: Compute translational velocities and travel direction in egocentric coordinates.
    ##########################################################################################################
            
    # Copy dx_ft and dy_ft to dx_fly and dy_fly
    posDf['dx_fly']   =   posDf['dx_ft']
    posDf['dy_fly']   =   posDf['dy_ft']
    
    # Compute the fly's egocentric translational speed (should be same as allocentric translational speed)
    posDf['vT_fly'] = np.hypot(posDf['dx_fly'], posDf['dy_fly']) * (1/posDf.dt) 
    
    # Compute the fly's forward and side velocities
    posDf['vF_fly']  =  posDf["dx_fly"] * (1/posDf.dt) 
    posDf['vL_fly']  =  posDf["dy_fly"] * (1/posDf.dt)  #should be +ve in the rightward direction
    
    # Optionally filter the above traces
    if ToFilter:
        posDf['dx_ft']         =  posDf['dx_ft'].rolling(window = BoxCarWidth, center = True,  min_periods=1).mean() 
        posDf['dy_ft']         =  posDf['dy_ft'].rolling(window = BoxCarWidth, center = True,  min_periods=1).mean()
        if 'dxattempt_ft' in posDf.columns:
            posDf['dxattempt_ft']  =  posDf['dxattempt_ft'].rolling( window=BoxCarWidth, center=True, min_periods=int(np.ceil(BoxCarWidth/2)) ).apply(np.nanmean, raw=True) 
            posDf['dyattempt_ft']  =  posDf['dyattempt_ft'].rolling (window=BoxCarWidth, center=True, min_periods=int(np.ceil(BoxCarWidth/2)) ).apply(np.nanmean, raw=True) 
        posDf['dx_fly']         =  posDf['dx_fly'].rolling(window = BoxCarWidth, center = True,  min_periods=1).mean() 
        posDf['dy_fly']         =  posDf['dy_fly'].rolling(window = BoxCarWidth, center = True,  min_periods=1).mean() 
        posDf['vT_fly']         =  posDf['vT_fly'].rolling(window = BoxCarWidth, center = True,  min_periods=1).mean() 
        posDf['vF_fly']         =  posDf['vF_fly'].rolling(window = BoxCarWidth, center = True,  min_periods=1).mean() 
        posDf['vL_fly']         =  posDf['vL_fly'].rolling(window = BoxCarWidth, center = True,  min_periods=1).mean() 
        
    # Lastly, compute the fly's egocentric translational angle. As above, we do this after filter because this is a circular variable. 
    posDf['vT_fly_angle']  =  np.arctan2( posDf['dy_fly'], posDf['dx_fly'] )/2/np.pi*360  # 0 degrees is y axis since arctan2(1,0) is 90 degrees. 
    #TODO: verify that this convention works with the zero defined by the brightest side.
    
    return posDf

def getTimeDf(uvrDat, trialDir, posDf = None, imaging = False, rate = 9.5509):
    
    if posDf is None: posDf = uvrDat.posDf
    
    if imaging:
    
        imgDat = pd.read_csv(sep.join([trialDir.replace('uvr','img'), 
        'roiDFF.csv'])).drop(columns =['Unnamed: 0'])

        with open(sep.join([trialDir.replace('uvr','img'),'imgMetadata.json'])) as json_file:
            imgMetadat = json.load(json_file)

        imgInd, volFramePos = align2img.findImgFrameTimes(uvrDat, imgMetadat)

        timeDf = align2img.combineImagingAndPosDf(imgDat, posDf, volFramePos)
        timeDf = timeDf.rename(columns = {'time [s]': 'time'})
    
    else:
        timeDf = posDf.iloc[::int(np.round(len(posDf['time'])/(rate*posDf['time'].max()))),:].reset_index().copy()
        timeDf['ds'] = np.diff(timeDf.s.values,prepend=0)
        timeDf['dx'] = np.diff(timeDf.x.values,prepend=0)
        timeDf['dy'] = np.diff(timeDf.y.values,prepend=0)
        timeDf = timeDf.drop(columns=['index'])
    
    timeDf['count'] = np.arange(1,len(timeDf['time'])+1,1)
    timeDf = carryAttrs(timeDf,posDf)
    
    return timeDf

def savgolFilterInterpolated(posDf, value, window, order, timeStr = 'time', kind='linear'):
        from scipy.signal import savgol_filter

        df = posDf.copy()

        #filtering with a savgol filter assumes that the data is evenly spaced
        #interpolating the data to be evenly spaced
        interpolated = sp.interpolate.interp1d(df[timeStr], df[value], kind=kind, bounds_error=False, fill_value='extrapolate')

        equalTime = np.linspace(df[timeStr].min(),df[timeStr].max(),2*len(df[timeStr]))

        filt = savgol_filter(interpolated(equalTime), window, order)

        filtInterpolated = sp.interpolate.interp1d(equalTime, filt, kind='nearest', bounds_error=False, fill_value='extrapolate')

        return filtInterpolated(df.time)