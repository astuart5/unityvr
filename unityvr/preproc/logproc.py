### This module contains basic preprocessing functions for processing the unity VR log file

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from os import mkdir, makedirs
from os.path import sep, isfile, exists
import json
import numpy as np
from scipy import interpolate
import warnings
from scipy.signal import medfilt

#dataframe column defs
objDfCols = ['name','collider','px','py','pz','rx','ry','rz','sx','sy','sz']

posDfCols = ['frame','time','x','y','angle']
ftDfCols = ['frame','ficTracTReadMs','ficTracTWriteMs','wx_ft','wy_ft','wz_ft']
dtDfCols = ['frame','time','dt']
tempDfCols = ['frame','tempReadTime','temperature']
nidDfCols = ['frame','time','dt','pdsig','imgfsig']
texDfCols = ['frame','time','xtex','ytex']
vidDfCols = ['frame','time','img','duration']
attmptDfCols = ['frame','time','dxattempt_ft','dyattempt_ft','angleattempt_ft']

# Data class definition

@dataclass
class unityVRexperiment:

    # metadata as dict
    metadata: dict

    imaging: bool = False
    brainregion: str = None

    # timeseries data
    posDf: pd.DataFrame = pd.DataFrame(columns=posDfCols)
    ftDf: pd.DataFrame = pd.DataFrame(columns=ftDfCols)
    nidDf: pd.DataFrame = pd.DataFrame(columns=nidDfCols)
    texDf: pd.DataFrame = pd.DataFrame(columns=texDfCols)
    vidDf: pd.DataFrame = pd.DataFrame(columns=vidDfCols)
    attmptDf: pd.DataFrame = pd.DataFrame(columns=attmptDfCols)
    shapeDf: pd.DataFrame = pd.DataFrame()
    timeDf: pd.DataFrame = pd.DataFrame()
    flightDf: pd.DataFrame = pd.DataFrame()
    tempDf: pd.DataFrame = pd.DataFrame()

    # object locations
    objDf: pd.DataFrame = pd.DataFrame(columns=objDfCols)

    # methods
    def printMetadata(self):
        print('Metadata:\n')
        for key in self.metadata:
            print(key, ' : ', self.metadata[key])

    ## data wrangling
    def downsampleftDf(self):
        frameftDf = self.ftDf.groupby("frame").sum()
        frameftDf.reset_index(level=0, inplace=True)
        return frameftDf

    def saveData(self, saveDir, saveName):
        savepath = sep.join([saveDir,saveName,'uvr'])

        # make directory
        if not exists(savepath):
            makedirs(savepath)

        # save metadata
        with open(sep.join([savepath,'metadata.json']), 'w') as outfile:
            json.dump(self.metadata, outfile,indent=4)

        # save dataframes
        self.objDf.to_csv(sep.join([savepath,'objDf.csv']))
        self.posDf.to_csv(sep.join([savepath,'posDf.csv']))
        self.ftDf.to_csv(sep.join([savepath,'ftDf.csv']))
        self.nidDf.to_csv(sep.join([savepath,'nidDf.csv']))
        self.texDf.to_csv(sep.join([savepath,'texDf.csv']))
        self.vidDf.to_csv(sep.join([savepath,'vidDf.csv']))
        self.attmptDf.to_csv(sep.join([savepath,'attmptDf.csv']))
        self.shapeDf.to_csv(sep.join([savepath,'shapeDf.csv']))
        self.timeDf.to_csv(sep.join([savepath,'timeDf.csv']))
        self.flightDf.to_csv(sep.join([savepath,'flightDf.csv']))
        self.tempDf.to_csv(sep.join([savepath,'tempDf.csv']))

        return savepath

# constructor for unityVRexperiment
def constructUnityVRexperiment(dirName,fileName,enforce_cm = False,colKeyPairs=None,**kwargs):

    dat = openUnityLog(dirName, fileName)

    metadat = makeMetaDict(dat, fileName)
    objDf = objDfFromLog(dat, enforce_cm=enforce_cm)
    posDf, ftDf, nidDf = timeseriesDfFromLog(dat, colKeyPairs=colKeyPairs, enforce_cm=enforce_cm, **kwargs)
    texDf = texDfFromLog(dat)
    vidDf = vidDfFromLog(dat)
    attmptDf = attmptDfFromLog(dat, enforce_cm=enforce_cm)
    tempDf = tempDfFromLog(dat)

    uvrexperiment = unityVRexperiment(metadata=metadat,posDf=posDf,ftDf=ftDf,nidDf=nidDf,objDf=objDf,texDf=texDf, vidDf=vidDf, attmptDf=attmptDf, tempDf=tempDf)

    return uvrexperiment


def loadUVRData(savepath):

    with open(sep.join([savepath,'metadata.json'])) as json_file:
        metadat = json.load(json_file)
    objDf = pd.read_csv(sep.join([savepath,'objDf.csv'])).drop(columns=['Unnamed: 0'])
    # ToDo remove when Shivam has removed fixation values from posDf
    posDf = pd.read_csv(sep.join([savepath,'posDf.csv']),dtype={'fixation': 'string'}).drop(columns=['Unnamed: 0','fixation'],errors='ignore')
    ftDf = pd.read_csv(sep.join([savepath,'ftDf.csv'])).drop(columns=['Unnamed: 0'])

    try: texDf = pd.read_csv(sep.join([savepath,'texDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        texDf = pd.DataFrame()
        #No texture mapping time series was recorded with this experiment, fill with empty DataFrame

    try: vidDf = pd.read_csv(sep.join([savepath,'vidDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        vidDf = pd.DataFrame()
        #No static images were displayed, fill with empty DataFrame

    try: attmptDf = pd.read_csv(sep.join([savepath,'attmptDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        attmptDf = pd.DataFrame()
        #no deviation between fictrac and unity

    try: shapeDf = pd.read_csv(sep.join([savepath,'shapeDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        shapeDf = pd.DataFrame()
        #Shape dataframe was not computed. Fill with empty DataFrame

    try: timeDf = pd.read_csv(sep.join([savepath,'timeDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        timeDf = pd.DataFrame()
        #Time dataframe was not computed. Fill with empty DataFrame
    try:
        flightDf = pd.read_csv(sep.join([savepath,'flightDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        flightDf = pd.DataFrame()
        #Flight dataframe was not computed. Fill with empty DataFrame
    
    try: nidDf = pd.read_csv(sep.join([savepath,'nidDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        nidDf = pd.DataFrame()
        #Nidaq dataframe may not have been extracted from the raw data due to memory/time constraints

    try: tempDf = pd.read_csv(sep.join([savepath,'tempDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        tempDf = pd.DataFrame()
        #No temperature time series was recorded with this experiment, fill with empty DataFrame

    uvrexperiment = unityVRexperiment(metadata=metadat,posDf=posDf,ftDf=ftDf,nidDf=nidDf,
                                      objDf=objDf,texDf=texDf,shapeDf=shapeDf,timeDf=timeDf, vidDf=vidDf, flightDf=flightDf, attmptDf=attmptDf, tempDf=tempDf)

    return uvrexperiment


def parseHeader(notes, headerwords, metadat):

    for i, hw in enumerate(headerwords[:-1]):
        if hw in notes:
            metadat[i] = notes[notes.find(hw)+len(hw)+1:notes.find(headerwords[i+1])].split('~')[0].strip()

    return metadat


def makeMetaDict(dat, fileName):
    headerwords = ["expid", "experiment", "genotype","flyid","sex","notes","temperature","\n"]
    metadat = ['testExp', 'test experiment', 'testGenotype', 'NA', 'NA', "NA", "NA"]

    if 'headerNotes' in dat[0].keys():
        headerNotes = dat[0]['headerNotes']
        metadat = parseHeader(headerNotes, headerwords, metadat)

    [datestr, timestr] = fileName.split('.')[0].split('_')[1:3]

    matching = [s for s in dat if "ficTracBallRadius" in s]
    if len(matching) == 0:
        print('no fictrac metadata')
        ballRad = 0.0
        translationalGain = 1.0
    else:
        ballRad = matching[0]["ficTracBallRadius"]
        try: translationalGain = matching[0]["translationalGain"]
        except: translationalGain = 1.0

    matching = [s for s in dat if "refreshRateHz" in s]
    setFrameRate = matching[0]["refreshRateHz"]

    metadata = {
        'expid': metadat[0],
        'experiment': metadat[1],
        'genotype': metadat[2],
        'sex': metadat[4],
        'flyid': metadat[3],
        'trial': 'trial'+fileName.split('.')[0].split('_')[-1][1:],
        'date': datestr,
        'time': timestr,
        'ballRad': ballRad,
        'translationalGain': translationalGain,
        'setFrameRate': setFrameRate,
        'notes': metadat[5],
        'temperature': metadat[6],
        'angle_convention':"right-handed"
    }

    return metadata


def openUnityLog(dirName, fileName):
    '''load json log file'''
    import json
    from os.path import sep

    # Opening JSON file
    f = open(sep.join([dirName, fileName]))

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    return data


# Functions for extracting data from log file and converting it to pandas dataframe

def objDfFromLog(dat, enforce_cm = False):
    # get dataframe with info about objects in vr
    matching = [s for s in dat if "meshGameObjectPath" in s]
    matchingRad = [s for s in dat if "ficTracBallRadius" in s]
    if 'translationalGain' in matchingRad[0]:
        gainVal = matchingRad[0]['translationalGain']
    else:
        gainVal = 1.0
    if gainVal == 0:
        warnings.warn('Translational gain is zero. Object sizes will not be modified.')
        gainVal = 1.0
    if enforce_cm:
        convf = 10.0
    else:
        convf = 1.0
    rows = []
    for match in matching:
        rows.append({'name': match['meshGameObjectPath'],
                    'collider': match['colliderType'],
                    'px': match['worldPosition']['x']/gainVal*convf,
                    'py': match['worldPosition']['z']/gainVal*convf,
                    'pz': match['worldPosition']['y'],
                    'rx': match['worldRotationDegs']['x'],
                    'ry': match['worldRotationDegs']['z'],
                    'rz': match['worldRotationDegs']['y'],
                    'sx': match['worldScale']['x']/gainVal*convf,
                    'sy': match['worldScale']['z']/gainVal*convf,
                    'sz': match['worldScale']['y']})
    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame()


def posDfFromLog(dat, posDfKey='attemptedTranslation', fictracSubject=None, ignoreKeys=['meshGameObjectPath'], enforce_cm = False):
    # get info about camera position in vr
    matching = [s for s in dat if ((posDfKey in s) & (np.all([i not in s for i in ignoreKeys])))] #checks key to extract from that particular dump
    matchingRad = [s for s in dat if "ficTracBallRadius" in s]
    if 'translationalGain' in matchingRad[0]:
        gainVal = matchingRad[0]['translationalGain']
    else:
        gainVal = 1.0
    if gainVal == 0:
        warnings.warn('Translational gain is zero. Fly remains stationary in the world.')
        gainVal = np.inf
    if enforce_cm:
        convf = 10.0
    else:
        convf = 1.0
    rows = []
    for match in matching:
        if fictracSubject != 'Integrated':
            rows.append({'frame': match['frame'],
                        'time': match['timeSecs'],
                        'x': match['worldPosition']['x']/gainVal*convf,
                        'y': match['worldPosition']['z']/gainVal*convf, #axes are named differently in Unity
                        'angle': (-match['worldRotationDegs']['y'])%360, #flip due to left handed convention in Unity
                        'dx_ft': match['actualTranslation']['x']/gainVal*convf,
                        'dy_ft': match['actualTranslation']['z']/gainVal*convf,
                        'dxattempt_ft': match['attemptedTranslation']['x']/gainVal*convf,
                        'dyattempt_ft': match['attemptedTranslation']['z']/gainVal*convf
                       })
        else:
            rows.append({'frame': match['frame'],
                            'time': match['timeSecs'],
                            'x': match['worldPosition']['x']/gainVal*convf,
                            'y': match['worldPosition']['z']/gainVal*convf, #axes are named differently in Unity
                            'angle': (-match['worldRotationDegs']['y'])%360, #flip due to left handed convention in Unity
                        })
    print('correcting for Unity angle convention.')

    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame()


def ftDfFromLog(dat):
    # get fictrac data
    matching = [s for s in dat if "ficTracDeltaRotationVectorLab" in s]
    rows = []
    for match in matching:
        rows.append({'frame': match['frame'],
                        'ficTracTReadMs': match['ficTracTimestampReadMs'],
                        'ficTracTWriteMs': match['ficTracTimestampWriteMs'],
                        'wx_ft': match['ficTracDeltaRotationVectorLab']['x'],
                        'wy_ft': match['ficTracDeltaRotationVectorLab']['y'],
                        'wz_ft': match['ficTracDeltaRotationVectorLab']['z']})

    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame()
    

def attmptDfFromLog(dat, enforce_cm = False):
    # get fictrac data during open loop periods
    matching = [s for s in dat if "fictracAttempt" in s]
    matchingRad = [s for s in dat if "ficTracBallRadius" in s]
    if enforce_cm:
        convf = 10.0
    else:
        convf = 1.0
    rows = []
    for match in matching:
        rows.append({'frame': match['frame'],
                    'time': match['timeSecs'],
                        'dyattempt_ft': -match['fictracAttempt']['x']*matchingRad[0]['ficTracBallRadius']*convf,
                        #scale by ball radius but not by translational gain to get true x,y in unity units (dm or if enforced cm), rightward motion
                        'dxattempt_ft': match['fictracAttempt']['y']*matchingRad[0]['ficTracBallRadius']*convf, #forward motion
                        'angleattempt_ft': (-np.rad2deg(match['fictracAttempt']['z']))%360}) #convert to degrees and flip to align with unity convention

    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame()


def dtDfFromLog(dat):
    # get delta time info
    matching = [s for s in dat if "deltaTime" in s]
    rows = []
    for match in matching:
        rows.append({'frame': match['frame'],
                    'time': match['timeSecs'],
                    'dt': match['deltaTime']})

    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame()

## function to load signals from NI-DAQ
"""def pdDfFromLog(dat, colKeyPairs={'imgFrameTrigger':'imgfsig', 'tracePD':'pdsig'}):
    # get NiDaq signal
    matching = [s for s in dat if any(key in s for key in colKeyPairs)]
    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        framedat = {'frame': match['frame'],
                        'time': match['timeSecs'],
                        'pdsig': match['tracePD'],
                        'imgfsig': match['imgFrameTrigger']}
        framedat = {'frame': match['frame'],
                        'time': match['timeSecs']}
        for key in colKeyPairs:
            if key in match:
                framedat[colKeyPairs[key]] = match[key]
            else:
                framedat[colKeyPairs[key]] = np.nan
        entries[entry] = pd.Series(framedat).to_frame().T

    if len(entries) > 0:
        #pdDf = pd.concat(entries,ignore_index = True)[['frame', 'time','imgfsig']]#.drop_duplicates()
        pdDf = pd.concat(entries,ignore_index = True)[['frame', 'time']+list(colKeyPairs.values())]#.drop_duplicates()
        return pdDf
    else:
        return pd.DataFrame()"""

def nidRawDfFromLog(dat, colKeyPairs={'imgFrameTrigger': 'imgfsig', 'tracePD': 'pdsig'}):
    # Extract raw NI-DAQ sampled signals from the log into a dataframe (before merging with frame timing)
    rows = []
    out_cols = ['frame', 'time'] + list(colKeyPairs.values())

    for match in dat:
        if not any(key in match for key in colKeyPairs):
            continue

        row = {
            'frame': match['frame'],
            'time': match['timeSecs'],
        }
        for key, out_key in colKeyPairs.items():
            row[out_key] = match.get(key, np.nan)

        rows.append(row)

    if rows:
        return pd.DataFrame(rows, columns=out_cols)
    else:
        return pd.DataFrame(columns=out_cols)


def texDfFromLog(dat):
    
    # get texture names
    matchingSessionParams = [s for s in dat if "sessionParameters" in s]
    #get texture names
    textureMatches = list(pd.Series([dict(l.split(':', 1) for l in matchingSessionParams[0]['sessionParameters']
    )[m] for m in dict(l.split(':', 1) for l in matchingSessionParams[0]['sessionParameters']
    ).keys() if 'Texture' in m]).replace(r"^\s*$", pd.NA, regex=True).dropna().str.split('\\').str[-1])

    # get texture remapping log
    matching = [s for s in dat if "xpos" in s]
    if len(matching) == 0: return pd.DataFrame()

    rows = []
    for entry, match in enumerate(matching):
        if 'ypos' in match:
            rows.append({'frame': match['frame'],
                        'time': match['timeSecs'],
                        'xtex': match['xpos'],
                        'ytex': match['ypos'],
                        'texName': textureMatches[entry%len(textureMatches)]
                        })
        else:
            rows.append({'frame': match['frame'],
                        'time': match['timeSecs'],
                        'xtex': match['xpos'],
                        'ytex': 0,
                        'texName': textureMatches[entry%len(textureMatches)]
                        })

    if rows:
        dtDf = dtDfFromLog(dat)
        texDf = pd.DataFrame(rows)
        texDf = pd.merge(dtDf, texDf, on=["frame", "time"], how='inner')
        texDf.time = texDf.time-texDf.time[0]
        return texDf[~texDf.duplicated(subset=['frame', 'texName'], keep='last')].reset_index(drop=True)
    else:
        return pd.DataFrame()


def vidDfFromLog(dat):
    # get static image presentations and times
    matching = [s for s in dat if "backgroundTextureNowInUse" in s]
    if len(matching) == 0: return pd.DataFrame()

    rows = []
    for match in matching:
        rows.append({'frame': match['frame'],
                    'time': match['timeSecs'],
                    'img': match['backgroundTextureNowInUse'].split('/')[-1],
                    'duration': match['durationSecs']})

    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame()
    

def tempDfFromLog(dat):
    # get static image presentations and times
    matching = [s for s in dat if "temperature" in s]
    if len(matching) == 0: return pd.DataFrame()

    rows = []
    for match in matching:
        rows.append({'frame': match['frame'],
                    'tempReadTime': match['timeSecs'],
                    'temperature': match['temperature']
                    })

    if rows:
        tempDf = pd.DataFrame(rows).groupby('frame').mean().reset_index() #average over multiple temperature readings per unity frame
        dtDf = dtDfFromLog(dat) #get the frame times
        if len(dtDf)>0: 
            tempDf = pd.merge(dtDf, tempDf, on="frame", how='outer')
            tempDf.time = tempDf.time-tempDf.time[0]
        return tempDf
    else:
        print('No temperature data was recorded.')
        return pd.DataFrame()


def ftTrajDfFromLog(directory, filename):
    cols = [14,15,16,17,18]
    colnames = ['x','y','heading','travelling','speed']
    ftTrajDf = pd.read_csv(directory+"/"+filename,usecols=cols,names=colnames)
    return ftTrajDf

def timeseriesDfFromLog(dat, colKeyPairs=None, **posDfKeyWargs):

    if colKeyPairs is None:
        colKeyPairs = {'imgFrameTrigger':'imgfsig', 'tracePD':'pdsig'}

    posDf = pd.DataFrame(columns=posDfCols)
    ftDf = pd.DataFrame(columns=ftDfCols)
    dtDf = pd.DataFrame(columns=dtDfCols)

    nidRawDf = pd.DataFrame(columns = ['frame', 'time']+list(colKeyPairs.values())) #['frame','time','pdsig','imgfsig']

    posDf = posDfFromLog(dat,**posDfKeyWargs)
    ftDf = ftDfFromLog(dat)
    dtDf = dtDfFromLog(dat)

    try: nidRawDf = nidRawDfFromLog(dat, colKeyPairs=colKeyPairs)
    except: print("No analog input data was recorded.")

    if len(posDf) > 0: posDf.time = posDf.time-posDf.time[0]
    if len(dtDf) > 0: dtDf.time = dtDf.time-dtDf.time[0]
    if len(nidRawDf) > 0: nidRawDf.time = nidRawDf.time-nidRawDf.time[0]

    if len(ftDf) > 0:
        ftDf.ficTracTReadMs = ftDf.ficTracTReadMs-ftDf.ficTracTReadMs[0]
        ftDf.ficTracTWriteMs = ftDf.ficTracTWriteMs-ftDf.ficTracTWriteMs[0]
    else:
        print("No fictrac signal was recorded.")

    if len(dtDf) > 0:
        posDf = pd.merge(dtDf, posDf, on="frame", how='outer').rename(columns={'time_x':'time'}).drop(['time_y'],axis=1)

    if len(nidRawDf) > 0 and len(dtDf) > 0:

        nidDf = pd.merge(dtDf, nidRawDf, on="frame", how='left').rename(columns={'time_x':'time'}).drop(['time_y'],axis=1)

        if 'pdsig' in nidDf.columns:
            nidDf["pdFilt"]  = nidDf.pdsig.values
            nidDf.pdFilt.values[np.isfinite(nidDf.pdsig.values)] = medfilt(nidDf.pdsig.values[np.isfinite(nidDf.pdsig.values)])
            #nidDf["pdThresh"]  = 1*(np.asarray(nidDf.pdFilt>=np.nanmedian(nidDf.pdFilt.values)))

        #nidDf["imgfFilt"]  = nidDf.imgfsig.values
        #nidDf.imgfFilt.values[np.isfinite(nidDf.imgfsig.values)] = medfilt(nidDf.imgfsig.values[np.isfinite(nidDf.imgfsig.values)])
        #nidDf["imgfThresh"]  = 1*(np.asarray(nidDf.imgfFilt.values>=np.nanmedian(nidDf.imgfFilt.values))).astype(np.int8)

        nidDf = generateInterTime(nidDf)
    else:
        nidDf = pd.DataFrame()

    return posDf, ftDf, nidDf


def generateInterTime(tsDf):
    

    tsDf['framestart'] = np.hstack([0,1*np.diff(tsDf.time)>0])
    #tsDf['framestart'] = tsDf.framestart.astype(bool)

    tsDf['counts'] = 1
    #tsDf['counts'] = tsDf.counts.astype(np.int8)
    sampperframe = tsDf.groupby('frame').sum()[['time','dt','counts']].reset_index(level=0).copy()
    sampperframe['fs'] = sampperframe.counts/sampperframe.dt

    frameStartIndx = np.hstack((0,np.where(tsDf.framestart)[0]))
    frameStartIndx = np.hstack((frameStartIndx, frameStartIndx[-1]+sampperframe.counts.values[-1]-1))
    frameIndx = tsDf.index.values
    #del sampperframe

    frameNums = tsDf.frame[frameStartIndx].values.astype('int')
    timeAtFramestart = tsDf.time[frameStartIndx].values

    #generate interpolated frames
    frameinterp_f = interpolate.interp1d(frameStartIndx,frameNums,bounds_error=False,fill_value='extrapolate')
    tsDf['frameinterp'] = frameinterp_f(frameIndx)

    timeinterp_f = interpolate.interp1d(frameStartIndx,timeAtFramestart,bounds_error=False,fill_value='extrapolate')
    tsDf['timeinterp'] = timeinterp_f(frameIndx)

    return tsDf


'''
def generateInterTime(tsDf):
    # Mark the start of each new frame
    tsDf['framestart'] = np.hstack([0, np.diff(tsDf.time) > 0])

    tsDf['counts'] = 1

    # Group by frame and calculate statistics per frame
    sampperframe = tsDf.groupby('frame').sum()[['time', 'dt', 'counts']].reset_index(level=0).copy()
    sampperframe['fs'] = sampperframe['counts'] / sampperframe['dt']

    # Get indices where a new frame starts
    frameStartIndx = np.hstack((0, np.where(tsDf.framestart)[0]))
    frameStartIndx = np.hstack((frameStartIndx, frameStartIndx[-1] + sampperframe['counts'].values[-1] - 1))
    frameIndx = tsDf.index.values

    # Frame and time interpolation
    frameNums = tsDf['frame'][frameStartIndx].values.astype('int')
    timeAtFramestart = tsDf['time'][frameStartIndx].values

    # Generate interpolated frame numbers
    frameinterp_f = interpolate.interp1d(frameStartIndx, frameNums, bounds_error=False, fill_value='extrapolate')
    tsDf['frameinterp'] = np.clip(frameinterp_f(frameIndx), frameNums[0], frameNums[-1])

    # Generate interpolated times
    timeinterp_f = interpolate.interp1d(frameStartIndx, timeAtFramestart, bounds_error=False, fill_value='extrapolate')
    tsDf['timeinterp'] = np.clip(timeinterp_f(frameIndx), timeAtFramestart[0], timeAtFramestart[-1])

    return tsDf
'''

# extract all dataframes from log file and save to disk
'''  # Not sure why this exists, will be deprecated
def convertJsonToPandas(dirName,fileName,saveDir, computePDtrace):

    dat = openUnityLog(dirName, fileName)
    metadat = makeMetaDict(dat, fileName)

    saveName = (metadat['expid']).split('_')[-1] + '/' + metadat['trial']
    savepath = sep.join([saveDir,saveName,'uvr'])

    # make directory
    if not exists(savepath): makedirs(savepath)

    # save metadata
    with open(sep.join([savepath,'metadata.json']), 'w') as outfile:
        json.dump(metadat, outfile,indent=4)

    # construct and save object dataframe
    objDf = objDfFromLog(dat)
    objDf.to_csv(sep.join([savepath,'objDf.csv']))

    # construct and save position and velocity dataframes
    posDf, ftDf, nidDf = timeseriesDfFromLog(dat, computePDtrace)
    posDf.to_csv(sep.join([savepath,'posDf.csv']))
    del posDf 
    ftDf.to_csv(sep.join([savepath,'ftDf.csv']))
    del ftDf 
    nidDf.to_csv(sep.join([savepath,'nidDf.csv']))
    del nidDf 

    # construct and save texture dataframes
    texDf = texDfFromLog(dat)
    vidDf = vidDfFromLog(dat)
    texDf.to_csv(sep.join([savepath,'texDf.csv']))
    del texDf 
    vidDf.to_csv(sep.join([savepath,'vidDf.csv']))
    del vidDf 

    return savepath
'''