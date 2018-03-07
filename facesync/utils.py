from __future__ import division

'''
    FaceSync Utils Class
    ==========================================
    VideoViewer: Watch video and plot data simultaneously.
    AudioAligner: Align two audios manually
    neutralface: points that show a face
    ChangeAU: change AUs and return new face
'''
__all__ = ['VideoViewer','AudioAligner','neutralface','audict','plotface','ChangeAU','read_facet']
__author__ = ["Jin Hyun Cheong"]
__license__ = "MIT"

import os
import numpy as np
import matplotlib.pyplot as plt



def read_facet(facetfile,fullfacet=False,demean = False,demedian=False,zscore=False,fillna=False,sampling_hz=None, target_hz=None):
    '''
    This function reads in an iMotions-FACET exported facial expression file. Uses downsample function from nltools.
    Arguments:
        fullfacet(def: False): If True, Action Units also provided in addition to default emotion predictions.
        demean(def: False): Demean data
        demedian(def: False): Demedian data
        zscore(def: False): Zscore data
        fillna(def: False): fill null values with ffill
        sampling_hz & target_hz: To downsample, specify the sampling hz and target hz.
    Returns:
        d: dataframe of processed facial expressions

    '''
    import pandas as pd

    def downsample(data,sampling_freq=None, target=None, target_type='samples', method='mean'):
        ''' Downsample pandas to a new target frequency or number of samples
            using averaging.
            Args:
                data: Pandas DataFrame or Series
                        sampling_freq:  Sampling frequency of data
                target: downsampling target
                        target_type: type of target can be [samples,seconds,hz]
                method: (str) type of downsample method ['mean','median'],
                        default: mean
            Returns:
                downsampled pandas object
        '''

        if not isinstance(data,(pd.DataFrame,pd.Series)):
            raise ValueError('Data must by a pandas DataFrame or Series instance.')
        if not (method=='median') | (method=='mean'):
            raise ValueError("Metric must be either 'mean' or 'median' ")

        if target_type is 'samples':
            n_samples = target
        elif target_type is 'seconds':
            n_samples = target*sampling_freq
        elif target_type is 'hz':
            n_samples = sampling_freq/target
        else:
            raise ValueError('Make sure target_type is "samples", "seconds", '
                            ' or "hz".')

        idx = np.sort(np.repeat(np.arange(1,data.shape[0]/n_samples,1),n_samples))
        # if data.shape[0] % n_samples:
        if data.shape[0] > len(idx):
            idx = np.concatenate([idx, np.repeat(idx[-1]+1,data.shape[0]-len(idx))])
        if method=='mean':
            return data.groupby(idx).mean().reset_index(drop=True)
        elif method=='median':
            return data.groupby(idx).median().reset_index(drop=True)

    d = pd.read_table(facetfile, skiprows=4, sep='\t',
                      usecols = ['FrameTime','Joy Evidence','Anger Evidence','Surprise Evidence','Fear Evidence','Contempt Evidence',
                      'Disgust Evidence','Sadness Evidence','Confusion Evidence','Frustration Evidence',
                      'Neutral Evidence','Positive Evidence','Negative Evidence','AU1 Evidence','AU2 Evidence',
                      'AU4 Evidence','AU5 Evidence','AU6 Evidence','AU7 Evidence','AU9 Evidence','AU10 Evidence',
                      'AU12 Evidence','AU14 Evidence','AU15 Evidence','AU17 Evidence','AU18 Evidence','AU20 Evidence',
                      'AU23 Evidence','AU24 Evidence','AU25 Evidence','AU26 Evidence','AU28 Evidence','AU43 Evidence','NoOfFaces',
                      'Yaw Degrees', 'Pitch Degrees', 'Roll Degrees'])
    # Choose index either FrameTime or FrameNo
    d = d.set_index(d['FrameTime'].values/1000.0)
    if type(fullfacet) == bool:
        if fullfacet==True:
            facets = ['Joy Evidence','Anger Evidence','Surprise Evidence','Fear Evidence','Contempt Evidence',
                      'Disgust Evidence','Sadness Evidence','Confusion Evidence','Frustration Evidence',
                      'Neutral Evidence','Positive Evidence','Negative Evidence','AU1 Evidence','AU2 Evidence',
                      'AU4 Evidence','AU5 Evidence','AU6 Evidence','AU7 Evidence','AU9 Evidence','AU10 Evidence',
                      'AU12 Evidence','AU14 Evidence','AU15 Evidence','AU17 Evidence','AU18 Evidence','AU20 Evidence',
                      'AU23 Evidence','AU24 Evidence','AU25 Evidence','AU26 Evidence','AU28 Evidence','AU43 Evidence','NoOfFaces',
                      'Yaw Degrees', 'Pitch Degrees', 'Roll Degrees']
        elif fullfacet == False:
            if type(fullfacet) == bool:
                facets = ['Joy Evidence','Anger Evidence','Surprise Evidence','Fear Evidence','Contempt Evidence',
                      'Disgust Evidence','Sadness Evidence','Confusion Evidence','Frustration Evidence',
                      'Neutral Evidence','Positive Evidence','Negative Evidence','NoOfFaces']
    else:
        facets = fullfacet
    d = d[facets] # change datatype to float16 for less memory use
    if zscore:
        d = (d.ix[:,:] - d.ix[:,:].mean()) / d.ix[:,:].std(ddof=0)
    if fillna:
        d = d.fillna(method='ffill')
    if demedian:
        d = d-d.median()
    if demean:
        d = d-d.mean()
    if sampling_hz and target_hz:
        d = downsample(d,sampling_freq=sampling_hz,target=target_hz,target_type='hz')
    return d


def rec_to_time(vals,fps):
    times = np.array(vals)/60./fps
    times = [str(int(np.floor(t))).zfill(2)+':'+str(int((t-np.floor(t))*60)).zfill(2) for t in times]
    return times

def VideoViewer(path_to_video, data_df,xlabel='', ylabel='',title='',figsize=(6.5,3),legend=False,xlim=None,ylim=None,plot_rows=False):
    """
    This function plays a video and plots the data underneath the video and moves a cursor as the video plays.
    Plays videos using Jupyter_Video_Widget by https://github.com/Who8MyLunch/Jupyter_Video_Widget
    Currently working on: Python 3
    For plot update to work properly plotting needs to be set to: %matplotlib notebook

    Args:
        path_to_video : file path or url to a video. tested with mov and mp4 formats.
        data_df : pandas dataframe with columns to be plotted in 30hz. (plotting too many column can slowdown update)
        ylabel(str): add ylabel
        legend(bool): toggle whether to plot legend
        xlim(list): pass xlimits [min,max]
        ylim(list): pass ylimits [min,max]
        plot_rows(bool): Draws individual plots for each column of data_df. (Default: True)
    """
    from jpy_video import Video
    from IPython.display import display, HTML
    display(HTML(data="""
    <style>
        div#notebook-container    { width: 95%; }
        div#menubar-container     { width: 65%; }
        div#maintoolbar-container { width: 99%; }
    </style>
    """))

    f = os.path.abspath(path_to_video)
    wid = Video(f)
    wid.layout.width='640px'
    wid.display()
    lnwidth = 3

    fps = wid.timebase**-1 # time base is play rate hard coded at 30fps
    print(fps)
    if plot_rows:
        fig,axs = plt.subplots(data_df.shape[1],1,figsize=figsize) # hardcode figure size for now..
    else:
        fig,axs = plt.subplots(1,1,figsize=figsize)
    t=wid.current_time
    if plot_rows and data_df.shape[1]>1:
        for ixs, ax in enumerate(axs):
            ax.axvline(fps*t,color='k',linestyle='--',linewidth=lnwidth) # cursor is always first of ax
            # plot each column
            data_df.iloc[:,ixs].plot(ax=ax,legend=legend,xlim=xlim,ylim=ylim)
            ax.set_xticks = np.arange(0,data_df.shape[0],5)
            ax.set(ylabel =data_df.columns[ixs], xlabel=xlabel, xticklabels = rec_to_time(ax.get_xticks(),fps))
    else:
        axs.axvline(fps*t,color='k',linestyle='--',linewidth=lnwidth) # cursor is always first of ax
        # plot each column
        data_df.plot(ax=axs,legend=legend,xlim=xlim,ylim=ylim)
        axs.set_xticks = np.arange(0,data_df.shape[0],5)
        axs.set(ylabel = data_df.columns[0],xlabel=xlabel, title=title, xticklabels = rec_to_time(axs.get_xticks(),fps))
    if legend:
        plt.legend(loc=1)
    plt.tight_layout()

    def plot_dat(axs,t,fps=fps):
        if plot_rows and data_df.shape[1]>1:
            for ax in axs:
                if ax.lines:
                    ax.lines[0].set_xdata([np.round(fps*t),np.round(fps*t)])
        else:
            if axs.lines:
                axs.lines[0].set_xdata([np.round(fps*t),np.round(fps*t)])
        fig.canvas.draw()

    def on_value_change(change,ax=axs,fps=fps):
        if change['name']=='_event':
            plot_dat(axs=axs, t=change['new']['currentTime'],fps=fps)

    #  call on_value_change that will call plotting function plot_dat whenever there is cursor update
    wid.observe(on_value_change)


def AudioAligner(original, sample, search_start=0.0,search_end=15.0, xmax = 60,manual=False,reduce_orig_volume=1):
    """
    This function pull up an interactive console to find the offsets between two audios.

    Args:
        original: path to original audio file (e.g. '../audios/original.wav')
        sample: path to the sample audio file (e.g. '../audios/sample.wav')
        search_start(float): start range for slider to search for offset
        search_end(float): end range for slider to search for offset
        xmax(int): Range of audio to plot from beginning
        manual(bool): set to True to turn off auto-refresh
        reduce_orig_volume(int or float): Original wav sounds are often larger so divide the volume by this number.
    """
    import scipy.io.wavfile as wav
    from IPython.display import Audio
    from IPython.display import display
    from ipywidgets import widgets

    orig_r,orig = wav.read(original)
    # volume is often louder on original so you can reduce it
    orig = orig/reduce_orig_volume
    # take one channel of target audio. probably not optimal
    if np.ndim(orig) >1:
        orig = orig[:,0]
    # grab one channel of sample audio
    tomatch_r,tomatch = wav.read(sample)
    if np.ndim(tomatch) >1:
        tomatch = tomatch[:,0]

    fs = 44100

    def audwidg(offset,play_start):
        allshift = play_start
        samplesize = 30
        tomatchcopy = tomatch[int((allshift+offset)*tomatch_r):int((allshift+offset)*tomatch_r)+fs*samplesize]
        shape = tomatchcopy.shape[0]
        origcopy = orig[int((allshift)*tomatch_r):int((allshift)*tomatch_r)+fs*samplesize]
        # when target audio is shorter, pad difference with zeros
        if origcopy.shape[0] < tomatchcopy.shape[0]:
            diff = tomatchcopy.shape[0] - origcopy.shape[0]
            origcopy = np.pad(origcopy, pad_width = (0,diff),mode='constant')
        toplay = origcopy + tomatchcopy
        display(Audio(data=toplay,rate=fs))

    def Plot_Audios(offset,x_min,x_max):
#         print('Precise offset : ' + str(offset))
        fig,ax = plt.subplots(figsize=(20,3))
        ax.plot(orig[int(fs*x_min):int(fs*x_max)],linewidth=.5,alpha=.8,color='r')
        ax.plot(tomatch[int(fs*x_min)+int(fs*offset) : int(fs*x_max)+int(fs*offset)],linewidth=.5,alpha=.8)
        ax.set_xticks([(tick-x_min)*fs for tick in range(int(x_min),int(x_max+1))])
        ax.set_xticklabels([tick for tick in range(int(x_min),int(x_max)+1)])
        ax.set_xlim([(x_min-x_min)*fs, (x_max-x_min)*fs] )
        ax.set_ylabel('Audio')
        ax.set_xlabel('Target Audio Time')
        audwidg(offset,x_min)
        plt.show()

    widgets.interact(Plot_Audios,
                     offset=widgets.FloatSlider(value = 0.5*(search_start+search_end), readout_format='.3f', min = float(search_start), max = float(search_end), step = 0.001,
                                                description='Adjusted offset: ',layout=widgets.Layout(width='90%')),
                     x_min=widgets.FloatSlider(description='Min X on audio plot', value=0.0,min=0.0,max=xmax,step=0.1, layout=widgets.Layout(width='50%')),
                     x_max=widgets.FloatSlider(description='Max X on audio plot', value=xmax,min=0.0,max=xmax,step=0.1, layout=widgets.Layout(width='50%')),
                     __manual=manual
                    )

neutralface = {-34: (212, 335),
 -33: (222, 342), -32: (237, 342), -30: (203, 335), -29: (222, 335),
 -28: (237, 328), -26: (227, 288), -25: (238, 292), -19: (201, 219),
 -18: (184, 220), -17: (169, 214), -16: (184, 204), -15: (201, 203),
 -14: (217, 215), -13: (225, 181), -12: (203, 172), -11: (180, 170),
 -10: (157, 174), -9: (142, 180), -8: (122, 222), -7: (126, 255),
 -6: (133, 286), -5: (139, 318), -4: (148, 349), -3: (165, 375),
 -2: (190, 397), -1: (219, 414),
 0: (252, 419),
 1: (285, 414), 2: (315, 398), 3: (341, 377), 4: (359, 351),
 5: (368, 319), 6: (371, 287), 7: (376, 254), 8: (378, 221),
 9: (354, 180), 10: (339, 173), 11: (316, 167), 12: (293, 171),
 13: (270, 180), 14: (281, 215), 15: (296, 203), 16: (314, 202),
 17: (328, 212), 18: (315, 219), 19: (297, 219), 20: (248, 207),
 21: (248, 227), 22: (248, 247), 23: (248, 268), 24: (248, 294),
 25: (260, 291), 26: (271, 287), 27: (248, 333), 28: (262, 328),
 29: (279, 335), 30: (296, 335), 31: (250, 340), 32: (264, 342),
 33: (280, 342), 34: (288, 335)}

audict = {'AU1' : {-11:(2,0),11:(-2,0),-12:(5,-8),12:(-5,-8),-13:(0,-20),13:(0,-20) },
# Brow Lowerer
'AU4': {-10:(4,5),10:(-4,5),-11:(4,15),11:(-4,15),-12:(5,20),12:(-5,20),-13:(0,15),13:(0,15) },
# Upper Lid Raiser
'AU5': {-9:(2,-9),9:(2,-9), -10:(2,-10),10:(-2,-10),-11:(2,-15),11:(-2,-15),
        -12:(5,-12),12:(-5,-12),-13:(0,-10),13:(0,-10),
          -16:(0,-10),-15:(0,-10),16:(0,-10),15:(0,-10),
          -19:(0,10),-18:(0,10),19:(0,10),18:(0,10)},
# cheek raiser
'AU6': {-8:(20,0),8:(-20,0), -7:(10,-5),7:(-10,-5), -6:(2,-8), 6:(-2,-8),
                   -9:(5,5),9:(-5,5),
                  17:(-5,5),18:(-3,-3),19:(-3,-3),
                  -17:(5,5),-18:(3,-3),-19:(3,-3)},
# nose wrinkler
'AU9': {-15:(2,4),15:(-2,4),-14:(2,3),14:(-2,3),
                    20:(0,5), 21:(0,-5), 22:(0,-7), 23:(0,-10),
                   -26:(5,-15),-25:(0,-15),24:(0,-15),25:(0,-15),26:(-5,-15),
                  -10:(2,0),10:(-2,0),-11:(2,8),11:(-2,8),
                  -12:(5,12),12:(-5,12),-13:(0,10),13:(0,10)
                  },
# Upper Lip Raiser
'AU10': {-34:(0,5),-33:(0,-2),-30:(0,3),-29:(0,-10),-28:(0,-5),
        -26:(-5,-8),-25:(0,-3),24:(0,-3),25:(0,-3),26:(5,-8),
        27:(0,-10),28:(0,-5),29:(0,-10),30:(0,3),33:(0,-2),34:(0,5)},
# Lip corner Puller
'AU12': { -30: (-10,-15), -34: (-5,-5), 30:(10,-15), 34:(5,-5),  -29:(0,0), 29:(0,0) },
#AU14 Dimpler
'AU14': {-33:(0,-5),-32:(0,-5),-30:(-5,-5),-28:(0,5),28:(0,5),30:(5,-5),31:(0,-5),32:(0,-5),33:(0,-5)},
# Chin raiser
'AU17': { -2:(5,0),-1:(5,-5),0:(0,-20),-1:(-5,-5),2:(-5,0)},
# Lip Puckerer
'AU18': {-30:(5,0), 30:(-5,0), -34:(5,0), 34:(-5,0),
        -33:(5,0),33:(-5,0), -29:(5,0),29:(-5,0),30:(-5,0),
        -28:(0,0),28:(0,0),27:(0,-8),31:(0,10),-32:(0,7),32:(0,7)} ,
# Lips Part
'AU25': {-28:(0,-3),28:(0,-3),27:(0,-5),31:(0,7),-32:(0,7),32:(0,7)},
# Lip Suck
'AU28': {-33:(0,-5),-32:(0,-5),-28:(0,5),24:(0,-3),28:(0,-5),31:(0,-5),32:(0,-5),33:(0,-5)}
 }

def plotface(face, scatter=True,line=False,annot=False,ax=None):
    """
    This function will take a dictionary of dots by (x,y) coordinates like the neutralface.

    """
    lineface = range(-8,9)
    linenose = list(range(20,24))
    linenose.extend([26,25,24,-25,-26,23])
    linelbrow = range(-13,-8)
    linerbrow = range(9,14)
    lineleye = list(range(-19,-13))
    lineleye.append(-19)
    linereye = list(range(14,20))
    linereye.append(14)
    linemouth = list(range(27,31))
    linemouth.extend([34,33,32,31,-32,-33,-34,-30,-29,-28,27])
    lines = [lineface,linenose,linelbrow,linerbrow,lineleye,linereye,linemouth]
    if not ax:
        f, ax = plt.subplots(1,1,figsize=(7,7))
    for key in face.keys():
        (x,y) = face[key]
        if scatter:
            ax.scatter(x,y,s=8,c='k')
        if annot:

            ax.annotate(key,(np.sign(key)*20+x,y))
        if line:
            for l in lines:
                ax.plot([face[key][0] for key in l],[face[key][1] for key in l],color='k' )
    ax.set_xlim([0,500])
    ax.set_ylim([0,500])
    ax.invert_yaxis()
    return ax

def ChangeAU(aulist, au_weight = 1.0, audict = audict, face = neutralface):
    '''
    This function will return a new face with the acti on units of aulist moved based on au_weight.

    Args:
        aulist: list of AUs that are activated currently supported include
        ['AU1','AU4','AU5','AU6','AU9', 'AU10', 'AU12','AU14','AU17','AU18','AU25','AU28']
        au_weights = float between 0 and 1.0 to activate all action unit or a
        dictionary to modular change of action units.
        audict = Dictionary of AU movements
        face = neutral face dictionary.
    '''
    au_weights = {}
    # if dict, apply weight to each au
    if type(au_weight)==dict:
        au_weights = au_weight
    # if a float apply to all
    elif type(au_weight)==float:
        for au in audict.keys():
            au_weights[au] = au_weight
    newface = face.copy()
    for au in aulist:
        for landmark in audict[au].keys():
            newface[landmark] = (face[landmark][0] + au_weights[au] * audict[au][landmark][0],
            face[landmark][1] + au_weights[au] *  audict[au][landmark][1])
    return newface
